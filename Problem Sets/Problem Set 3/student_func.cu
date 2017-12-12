#include <stdio.h>

/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

__global__
void minimum(float *d_min_logLum, const float* const d_logLuminance){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int tidx = threadIdx.x;

    __shared__
        float smem[1024];

    smem[tidx] = d_logLuminance[tid];
    __syncthreads();

    int s = blockDim.x / 2;

    while(s >= 1){
        if(tidx < s){
            smem[tidx] = min(smem[tidx] , smem[tidx + s]);
        }
        __syncthreads();
        s >>= 1;
    }

    if(tidx == 0){
        d_min_logLum[blockIdx.x] = smem[0];
    }
}


__global__
void maximum(float *d_max_logLum, const float* const d_logLuminance){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int tidx = threadIdx.x;

    __shared__
        float smem[1024];

    smem[tidx] = d_logLuminance[tid];
    __syncthreads();

    int s = blockDim.x / 2;

    while(s >= 1){
        if(tidx < s){
            smem[tidx] = max(smem[tidx], smem[tidx + s]);
        }
        __syncthreads();
        s >>= 1;
    }

    if(tidx == 0){
        d_max_logLum[blockIdx.x] = smem[0];
    }
}
__global__
void hist(unsigned int* const d_hist, const float* const d_logLuminance, const float min_logLum, const float range, const int numBins, const int size){

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if(tid < size){
        int bin = min(numBins - 1.0, (numBins * (d_logLuminance[tid] - min_logLum) / range));

        atomicAdd(&(d_hist[bin]), 1);
    }
}

__global__
void hillis_steele(unsigned int* const d_cdf, const unsigned int* const d_hist, int n){
    int tidx = threadIdx.x;

    if(tidx < n){
        __shared__
            unsigned int smem[2 * 1024];

        int head_in = 1;
        int head_out = 0;

        // right shift input, and store it in shared memory
        if(tidx == 0){
            smem[tidx] = 0;
        }
        else{
            smem[head_out * n + tidx] = d_hist[tidx - 1];
        }

        int offset = 1;

        while(offset < n){
            // swap buffer indices
            head_in = 1 - head_in;
            head_out = 1 - head_in;

            if(tidx < offset){
                smem[head_out * n + tidx] = smem[head_in * n + tidx];
            }else{
                smem[head_out * n + tidx] = smem[head_in * n + tidx] + smem[head_in * n + tidx - offset];
            }
            __syncthreads();

            offset <<= 1;
        }

        d_cdf[tidx] = smem[head_out * n + tidx];
    }
}

//uses 1024 threads
__global__
void blelloch1(unsigned int* const d_cdf, const unsigned int* const d_hist, int n){
    int tidx = threadIdx.x;

    if(tidx < n){
        int offset = 1;

        __shared__
            int smem[1024];

        smem[tidx] =  d_hist[tidx];
        __syncthreads();

        while(offset < n){
            bool condition = ((tidx + 1) % (2 * offset)) == 0;
            
            if(condition){
                smem[tidx] += smem[tidx - offset];
            }    
            __syncthreads();

            offset <<= 1;
        }

        if(tidx == n - 1){
            smem[tidx] = 0;
        }
        __syncthreads();

        offset = n >> 1;

        while(offset >= 1){
            bool condition = ((tidx + 1) % (2 * offset)) == 0;
            if(condition){
                int temp = smem[tidx - offset];
                smem[tidx - offset] = smem[tidx];
                smem[tidx] += temp;
            }
            __syncthreads();
            offset >>= 1;
        }

        d_cdf[tidx] = smem[tidx];
    }
}

// uses 512 threads
__global__
void blelloch2(unsigned int* const d_cdf, const unsigned int* const d_hist, int n){
    int tidx = threadIdx.x;
    int idx = 2 * tidx + 1;

    if(idx < n){
        int offset = 1;

        __shared__
            int smem[1024];

        smem[idx]     =  d_hist[idx];
        smem[idx - 1] =  d_hist[idx - 1];
        __syncthreads();

        while(offset < n){
            bool condition = ((idx + 1) % (2 * offset)) == 0;

            if(condition){
                smem[idx] += smem[idx - offset];
            }    
            __syncthreads();

            offset <<= 1;
        }

        if(idx == n - 1){
            smem[idx] = 0;
        }
        __syncthreads();

        offset = n >> 1;

        while(offset >= 1){
            bool condition = ((idx + 1) % (2 * offset)) == 0;
            if(condition){
                int temp = smem[idx - offset];
                smem[idx - offset] = smem[idx];
                smem[idx] += temp;
            }
            __syncthreads();
            offset >>= 1;
        }

        d_cdf[idx]     = smem[idx];
        d_cdf[idx - 1] = smem[idx - 1];
    }
}

// Parallel Prefix sum: Mark Harris
// uses 512 threads
__global__
void blelloch3(unsigned int* const d_cdf, const unsigned int* const d_hist, const int n){
  int tidx = threadIdx.x;
    
  if(tidx < n/2){
    __shared__
    int smem[1024];
     
    smem[ 2 * tidx ]     =  d_hist[ 2 * tidx ];
    smem[ 2 * tidx + 1 ] =  d_hist[ 2 * tidx + 1 ];
    __syncthreads();

    int d = n >> 1;
    int offset = 1;
    
    while(d > 0){
        if(tidx < d){
            int idx = offset * (2 * tidx + 2) - 1;
            smem[idx] += smem[idx - offset];
        }
        __syncthreads();

        offset <<= 1;
        d >>= 1;
    }

    if(tidx == 0){
      smem[n - 1] = 0;
    }
    __syncthreads();

    offset = n >> 1;
    d = 1;

    while(d < n){
        if(tidx < d){
            int idx = offset * (2 * tidx + 2) - 1;
            int temp = smem[idx - offset];
            smem[idx - offset] = smem[idx];
            smem[idx] += temp;
        }
        __syncthreads();
        offset >>= 1;
        d <<= 1;
    }

    d_cdf[2 * tidx]     = smem[2 * tidx];
    d_cdf[2 * tidx + 1] = smem[2 * tidx + 1];     
  }
}

void find_min(float *d_min_logLum, float *d_intermediate, const float* const d_logLuminance , const size_t size){
  const int maxThreadsPerBlock = 1024;

  int threads = maxThreadsPerBlock;
  int blocks = size / maxThreadsPerBlock;

  minimum<<<blocks, threads>>>(d_intermediate, d_logLuminance);

  blocks = 1;

  minimum<<<blocks, threads>>>(d_min_logLum, d_intermediate);
}

void find_max(float *d_max_logLum, float *d_intermediate, const float* const d_logLuminance, const size_t size){
  const int maxThreadsPerBlock = 1024;
  
  int threads = maxThreadsPerBlock;
  int blocks = size / maxThreadsPerBlock;

  maximum<<<blocks, threads>>>(d_intermediate, d_logLuminance);

  blocks = 1;

  maximum<<<blocks, threads>>>(d_max_logLum, d_intermediate);
}

void find_hist(unsigned int* const d_hist, const float* const d_logLuminance, const float min_logLum, const float range, const int numBins, const int size){
  int threads = 1024;
  int blocks = size/threads;

  hist<<<blocks, threads>>>(d_hist, d_logLuminance, min_logLum, range, numBins, size);
}

void prefix_sum(unsigned int* const d_cdf, const unsigned int* const d_hist, const int numBins){
  int threads = numBins;
  int blocks = 1;
  
  //hillis_steele<<<blocks, threads>>>(d_cdf, d_hist, numBins);
  //blelloch1<<<blocks, threads/2>>>(d_cdf, d_hist, numBins);
  blelloch2<<<blocks, threads/2>>>(d_cdf, d_hist, numBins);
  //blelloch3<<<blocks, threads/2>>>(d_cdf, d_hist, numBins);
}


#include "utils.h"

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */

  const size_t size = numRows * numCols;

  float *d_min_logLum;
  float *d_max_logLum;
  
  float range;

  float* d_intermediate;

  unsigned int *d_hist;

  unsigned int *h_hist = (unsigned int *)malloc(sizeof(unsigned int) * numBins);

  checkCudaErrors(cudaMalloc(&d_min_logLum, sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_max_logLum,  sizeof(float)));
  
  checkCudaErrors(cudaMalloc(&d_intermediate,  size * sizeof(float)));
  
  checkCudaErrors(cudaMalloc(&d_hist, sizeof(unsigned int) * numBins));
  checkCudaErrors(cudaMemset(d_hist, 0, sizeof(unsigned int) * numBins));
  
  find_min(d_min_logLum, d_intermediate, d_logLuminance, size);
  find_max(d_max_logLum, d_intermediate, d_logLuminance, size);
  
  checkCudaErrors(cudaMemcpy(&min_logLum, d_min_logLum, sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(&max_logLum, d_max_logLum, sizeof(float), cudaMemcpyDeviceToHost));

  range = max_logLum - min_logLum;

  find_hist(d_hist, d_logLuminance, min_logLum, range, numBins, size);
  checkCudaErrors(cudaMemcpy(h_hist, d_hist, sizeof(unsigned int) * numBins, cudaMemcpyDeviceToHost));

  prefix_sum(d_cdf, d_hist, numBins);
  
  /* Check cpu results: hist, cdf
  int sum = 0;
  for(int i = 0; i < 1024; ++i){
    sum += h_hist[i];
    printf("%d ", h_hist[i]);
  }
  printf("\nsize: %d, sum: %d\n", size, sum);

  int cdf_cpu[1024] = {0};

  int acc = 0;
  for(int i = 0; i < 1024; ++i){
    printf("%d ", acc);
    cdf_cpu[i] = acc;
    acc += h_hist[i];
  }
  printf("\n");
  */
  
  checkCudaErrors(cudaFree(d_min_logLum));
  checkCudaErrors(cudaFree(d_max_logLum));
}
