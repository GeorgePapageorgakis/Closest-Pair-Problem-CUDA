/* Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */

#include <assert.h>
#include <cuda.h>
#include "sortingNetworks_common.h"
#include "sortingNetworks_common.cuh"
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Monolithic bitonic sort kernel for short arrays fitting into shared memory
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void bitonicSortShared(
    float2 *d_P_out,
    float2 *d_P_in,
    uint arrayLength,
    uint dir,
	uint xy 
)
{
    //Shared memory storage for one or more short vectors
    __shared__ float2 s_key[SHARED_SIZE_LIMIT];

    //Offset to the beginning of subbatch and load data
    d_P_in  += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_P_out += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    s_key[threadIdx.x +                       0] = d_P_in[                      0];
    s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = d_P_in[(SHARED_SIZE_LIMIT / 2)];

    for (uint size = 2; size < arrayLength; size <<= 1) {
        //Bitonic merge
        uint ddd = dir ^ ((threadIdx.x & (size / 2)) != 0);

        for (uint stride = size / 2; stride > 0; stride >>= 1) {
            __syncthreads();
            uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            Comparator(	s_key[pos +  0], s_key[pos + stride], ddd, xy );
        }
    }

    //ddd == dir for the last bitonic merge step
    {
        for (uint stride = arrayLength / 2; stride > 0; stride >>= 1) {
            __syncthreads();
            uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            Comparator(	s_key[pos +  0], s_key[pos + stride], dir, xy );
        }
    }
	__syncthreads();
    d_P_out[                      0] = s_key[threadIdx.x +                       0];
    d_P_out[(SHARED_SIZE_LIMIT / 2)] = s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Bitonic sort kernel for large arrays (not fitting into shared memory)
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Bottom-level bitonic sort
//Almost the same as bitonicSortShared with the exception of
//even / odd subarrays being sorted in opposite directions
//Bitonic merge accepts both
//Ascending | descending or descending | ascending sorted pairs
__global__ void bitonicSortShared1(
    float2 *d_P_out,
    float2 *d_P_in,
	uint xy 
)
{
    //Shared memory storage for current subarray
    __shared__ float2 s_key[SHARED_SIZE_LIMIT];

    //Offset to the beginning of subarray and load data
    d_P_in  += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_P_out += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    s_key[threadIdx.x +                       0] = d_P_in[                      0];
    s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = d_P_in[(SHARED_SIZE_LIMIT / 2)];

    for (uint size = 2; size < SHARED_SIZE_LIMIT; size <<= 1) {
        //Bitonic merge
        uint ddd = (threadIdx.x & (size / 2)) != 0;

        for (uint stride = size / 2; stride > 0; stride >>= 1) {
            __syncthreads();
            uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            Comparator(	s_key[pos + 0], s_key[pos + stride], ddd, xy );
        }
    }

    //Odd / even arrays of SHARED_SIZE_LIMIT elements
    //sorted in opposite directions
    uint ddd = blockIdx.x & 1;
    {
        for (uint stride = SHARED_SIZE_LIMIT / 2; stride > 0; stride >>= 1) {
            __syncthreads();
            uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            Comparator(	s_key[pos + 0], s_key[pos + stride], ddd, xy );
        }
    }
    __syncthreads();
    d_P_out[                      0] = s_key[threadIdx.x +                       0];
    d_P_out[(SHARED_SIZE_LIMIT / 2)] = s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
}

//Bitonic merge iteration for stride >= SHARED_SIZE_LIMIT
__global__ void bitonicMergeGlobal(
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
    uint        comparatorI = global_comparatorI & (arrayLength / 2 - 1);

    //Bitonic merge
    uint ddd = dir ^ ((comparatorI & (size / 2)) != 0);
    uint pos = 2 * global_comparatorI - (global_comparatorI & (stride - 1));

    float2 keyA = d_P_in[pos +      0];
    float2 keyB = d_P_in[pos + stride];

    Comparator( keyA, keyB, ddd, xy );

    d_P_out[pos +      0] = keyA;
    d_P_out[pos + stride] = keyB;
}

//Combined bitonic merge steps for
//size > SHARED_SIZE_LIMIT and stride = [1 .. SHARED_SIZE_LIMIT / 2]
__global__ void bitonicMergeShared(
    float2 *d_P_out,
    float2 *d_P_in,
    uint arrayLength,
    uint size,
    uint dir,
	uint xy
)
{
    //Shared memory storage for current subarray
    __shared__ float2 s_key[SHARED_SIZE_LIMIT];

    d_P_in  += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_P_out += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    s_key[threadIdx.x +                       0] = d_P_in[                      0];
    s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = d_P_in[(SHARED_SIZE_LIMIT / 2)];

    //Bitonic merge
    uint comparatorI = UMAD(blockIdx.x, blockDim.x, threadIdx.x) & ((arrayLength / 2) - 1);
    uint ddd = dir ^ ((comparatorI & (size / 2)) != 0);

    for (uint stride = SHARED_SIZE_LIMIT / 2; stride > 0; stride >>= 1) {
        __syncthreads();
        uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
        Comparator( s_key[pos + 0], s_key[pos + stride], ddd, xy );
    }
    __syncthreads();
    d_P_out[                      0] = s_key[threadIdx.x +                       0];
    d_P_out[(SHARED_SIZE_LIMIT / 2)] = s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Interface function
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Helper function (also used by odd-even merge sort)
extern "C" uint factorRadix2(uint *log2L, uint L)
{
    if (!L) {
        *log2L = 0;
        return 0;
    }
    else {
        for (*log2L = 0; (L & 1) == 0; L >>= 1, *log2L++);
		return L;
    }
}

extern "C" uint bitonicSort(
    float2 *d_P_out,
    float2 *d_P_in,
    uint batchSize,
    uint arrayLength,
    uint dir,
	uint xy //1 for y,  0 for x
)
{
    //Nothing to sort
    if (arrayLength < 2)
        return 0;

    //Only power-of-two array lengths are supported by this implementation
    uint log2L;
    uint factorizationRemainder = factorRadix2(&log2L, arrayLength);
    assert(factorizationRemainder == 1);

    dir = (dir != 0);

    uint  blockCount = batchSize * arrayLength / SHARED_SIZE_LIMIT;
    uint threadCount = SHARED_SIZE_LIMIT / 2;

    if (arrayLength <= SHARED_SIZE_LIMIT) {
		//Kernel Call length fitting in Shared Memory
        assert((batchSize * arrayLength) % SHARED_SIZE_LIMIT == 0);
        bitonicSortShared<<<blockCount, threadCount>>>(d_P_out, d_P_in, arrayLength, dir, xy);
    }
    else {
        bitonicSortShared1<<<blockCount, threadCount>>>(d_P_out,  d_P_in, xy);
		//Kernel Call length NOT fitting in Shared Memory
        for (uint size = 2 * SHARED_SIZE_LIMIT; size <= arrayLength; size <<= 1){
            for (uint stride = size / 2; stride > 0; stride >>= 1){
                if (stride >= SHARED_SIZE_LIMIT) {
                    bitonicMergeGlobal<<<(batchSize * arrayLength) / 512, 256>>>(d_P_out, d_P_out, arrayLength, size, stride, dir, xy);
                }
                else {
                    bitonicMergeShared<<<blockCount, threadCount>>>(d_P_out, d_P_out, arrayLength, size, dir, xy);
                    break;
                }
			}
		}
    }
    return threadCount;
}

/*extern uint log_2(uint N){  
    return log( N ) / log( 2. );//. to avoid compiler complaining about ambigous call
}*/