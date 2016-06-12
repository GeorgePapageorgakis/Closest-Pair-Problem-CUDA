/* Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */

#include <assert.h>
#include <cuda.h>
#include "helper_cuda.h"
#include "sortingNetworks_common.h"
#include "sortingNetworks_common.cuh"

////////////////////////////////////////////////////////////////////////////////
// Monolithic Bacther's sort kernel for short arrays fitting into shared memory
////////////////////////////////////////////////////////////////////////////////
__global__ void oddEvenMergeSortShared(
    float2 *d_P_out,
    float2 *d_P_in,
    uint arrayLength,
    uint dir,
	uint xy
)
{
    //Shared memory storage for one or more small vectors
    __shared__ float2 s_key[SHARED_SIZE_LIMIT];

    //Offset to the beginning of subbatch and load data
    d_P_in  += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_P_out += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    s_key[threadIdx.x +                       0] = d_P_in[                      0];
    s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = d_P_in[(SHARED_SIZE_LIMIT / 2)];

    for (uint size = 2; size <= arrayLength; size <<= 1)
    {
        uint stride = size / 2;
        uint offset = threadIdx.x & (stride - 1);

        {
            __syncthreads();
            uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            Comparator( s_key[pos + 0], s_key[pos + stride], dir, xy );
            stride >>= 1;
        }

        for (; stride > 0; stride >>= 1) {
            __syncthreads();
            uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));

            if (offset >= stride)
                Comparator( s_key[pos - stride], s_key[pos + 0], dir, xy );
        }
    }
    __syncthreads();
    d_P_out[                      0] = s_key[threadIdx.x +                       0];
    d_P_out[(SHARED_SIZE_LIMIT / 2)] = s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
}


////////////////////////////////////////////////////////////////////////////////
// Odd-even merge sort iteration kernel
// for large arrays (not fitting into shared memory)
////////////////////////////////////////////////////////////////////////////////
__global__ void oddEvenMergeGlobal(
    float2 *d_P_out,
    float2 *d_P_in,
    uint arrayLength,
    uint size,
    uint stride,
    uint dir,
	uint xy
)
{
    uint global_comparatorI = blockIdx.x * blockDim.x + threadIdx.x;

    //Odd-even merge
    uint pos = 2 * global_comparatorI - (global_comparatorI & (stride - 1));

    if (stride < size / 2) {
        uint offset = global_comparatorI & ((size / 2) - 1);

        if (offset >= stride) {
            float2 keyA = d_P_in[pos - stride];
            float2 keyB = d_P_in[pos +      0];

            Comparator(keyA, keyB, dir, xy );

            d_P_out[pos - stride] = keyA;
            d_P_out[pos +      0] = keyB;
        }
    }
    else {
        float2 keyA = d_P_in[pos +      0];
        float2 keyB = d_P_in[pos + stride];

        Comparator( keyA, keyB, dir, xy );

        d_P_out[pos +      0] = keyA;
        d_P_out[pos + stride] = keyB;
    }
}



////////////////////////////////////////////////////////////////////////////////
// Interface function
////////////////////////////////////////////////////////////////////////////////
//Helper function
extern "C" uint factorRadix2(uint *log2L, uint L);

extern "C" void oddEvenMergeSort(
    float2 *d_P_out,
    float2 *d_P_in,
    uint batchSize,
    uint arrayLength,
    uint dir,
	uint xy
)
{
    //Nothing to sort
    if (arrayLength < 2)
        return;

    //Only power-of-two array lengths are supported by this implementation
    uint log2L;
    uint factorizationRemainder = factorRadix2(&log2L, arrayLength);
    assert(factorizationRemainder == 1);

    dir = (dir != 0);

    uint  blockCount = (batchSize * arrayLength) / SHARED_SIZE_LIMIT;
    uint threadCount = SHARED_SIZE_LIMIT / 2;

    if (arrayLength <= SHARED_SIZE_LIMIT) {
        assert(SHARED_SIZE_LIMIT % arrayLength == 0);
        oddEvenMergeSortShared<<<blockCount, threadCount>>>(d_P_out, d_P_in, arrayLength, dir, xy);
    }
    else {
        oddEvenMergeSortShared<<<blockCount, threadCount>>>(d_P_out, d_P_in, SHARED_SIZE_LIMIT, dir, xy);

        for (uint size = 2 * SHARED_SIZE_LIMIT; size <= arrayLength; size <<= 1)
            for (unsigned stride = size / 2; stride > 0; stride >>= 1) {
                //Unlike with bitonic sort, combining bitonic merge steps with
                //stride = [SHARED_SIZE_LIMIT / 2 .. 1] seems to be impossible as there are
                //dependencies between data elements crossing the SHARED_SIZE_LIMIT borders
                oddEvenMergeGlobal<<<(batchSize * arrayLength) / 512, 256>>>(d_P_out, d_P_out, arrayLength, size, stride, dir, xy);
            }
    }
}
