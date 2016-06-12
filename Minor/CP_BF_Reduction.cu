#include <stdio.h>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "cuComplex.h"
#include "device_functions.h"
#define N 1048576/2
#define BLOCK_SIZE 128
#define BLOCKS (N + (BLOCK_SIZE - 1))/BLOCK_SIZE

using namespace std;

__global__ void compare_float2s_BF( 
	unsigned long long *dev_count, 
	float2	*dev_P,
	float	*d_Dist, 
	float4	*d_Pairs
)
{
	unsigned int tid	= threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int stride = blockDim.x * gridDim.x;
	unsigned int cache_index, current_block, grid_offset = 0;
	float	local_dist, local_min;
	float2	temp;
	float4	local_ClosestPair;

	__shared__ float2 cache_P		[BLOCK_SIZE];
	__shared__ float  cache_d_Dist	[BLOCK_SIZE];
	__shared__ float4 cache_d_Pairs	[BLOCK_SIZE];
	__syncthreads();

	while (tid < N){
		temp.x				= dev_P[tid].x;
		temp.y				= dev_P[tid].y;
		local_min			= FLT_MAX;

		//cached blocks of P array stored in shared memory as cache[], current_block corresponds to P's cached sub-array
		for ( current_block = blockIdx.x + (grid_offset * gridDim.x); current_block < BLOCKS; current_block++){
			//check for boundaries violation within each current block
			if (current_block * blockDim.x + threadIdx.x < N){
				//fetch data from GLOBAL memory to SHARED memory, coalesced memory access
				cache_P[threadIdx.x] = dev_P[current_block * blockDim.x + threadIdx.x];
			}
			//synchronize threads in this block
			__syncthreads();

			//get the beginning of the cached block or the next point if it is a comparison on the same block, check boundaries of index
			for(current_block == blockIdx.x + (grid_offset * gridDim.x) ? cache_index = threadIdx.x + 1 : cache_index = 0; 
					(cache_index < blockDim.x) && (current_block * blockDim.x + cache_index) < N;
						cache_index++){
				//calculate distance of current points
				local_dist = (cache_P[cache_index].x - temp.x) * (cache_P[cache_index].x - temp.x) +
								(cache_P[cache_index].y - temp.y) * (cache_P[cache_index].y - temp.y);
				//Determine the minimum numeric value of the arguments
				if (local_dist < local_min){
					local_min = local_dist;
					local_ClosestPair.x = temp.x;
					local_ClosestPair.y = temp.y;
					local_ClosestPair.z = cache_P[cache_index].x;
					local_ClosestPair.w = cache_P[cache_index].y;
				}
			}
			// Synchronize to make sure that the preceding computation
            // is done before loading new elements in the next iteration
			__syncthreads();
		}
		cache_d_Dist	[threadIdx.x] =	local_min;
		cache_d_Pairs	[threadIdx.x] =	local_ClosestPair;
		__syncthreads();

		// Parallel Reduction: Sequential Addressing in shared mem (bitwise right shift i)
		// reversed loop and threadID-based indexing
		for (unsigned int i = blockDim.x / 2; i > 0; i >>= 1){
			if ((threadIdx.x < i) && (cache_d_Dist[threadIdx.x] > cache_d_Dist[threadIdx.x + i])){
				cache_d_Dist [threadIdx.x] = cache_d_Dist [threadIdx.x + i];
				cache_d_Pairs[threadIdx.x] = cache_d_Pairs[threadIdx.x + i];
			}
			__syncthreads();
		}
		// We now have the min value of each block stored in cache_CP[0] and 
		// we can store it in the corresponding dev_Closest_Pair[] for later global memory reduction;
		if (threadIdx.x == 0){
			d_Dist	[tid/BLOCK_SIZE] = cache_d_Dist [0];
			d_Pairs	[tid/BLOCK_SIZE] = cache_d_Pairs[0];
		}
		//for Arbitrarily P[] Length
		tid += stride;
		grid_offset ++;
		__syncthreads();
	}
}

__global__ void MinDistReduction(
	float   *d_Dist,
	float4  *d_Pairs
)
{
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int stride = blockDim.x;
	float local_d_Dist;
	float4 local_d_Pairs;

	__shared__ float  cache_d_Dist [BLOCK_SIZE];
	__shared__ float4 cache_d_Pairs[BLOCK_SIZE];
	__syncthreads();

	//fetch data from GLOBAL memory to SHARED memory, coalesced memory access
	local_d_Dist  = d_Dist [tid];
	local_d_Pairs = d_Pairs[tid];
	//cache_d_Dist [threadIdx.x] = d_Dist [tid];
	//cache_d_Pairs[threadIdx.x] = d_Pairs[tid];
	//initialize local_ClosestPair
	//local_d_Dist  = cache_d_Dist [threadIdx.x];
	//local_d_Pairs = cache_d_Pairs[threadIdx.x];
	tid += stride;
	__syncthreads();

	while (tid < N/BLOCK_SIZE){
		//Each thread compares its local element with the next +stride elements in d_Dist[].
		if (d_Dist[tid] < local_d_Dist){
			local_d_Dist  = d_Dist [tid];
			local_d_Pairs = d_Pairs[tid];
		}
		tid += stride;
		__syncthreads();
	}
	cache_d_Dist [threadIdx.x] = local_d_Dist;
	cache_d_Pairs[threadIdx.x] = local_d_Pairs;
	__syncthreads();

	// Parallel Reduction: Sequential Addressing in shared mem (bitwise right shift i) 
	// reversed loop and threadID-based indexing
	for(unsigned int i = blockDim.x / 2; i > 0; i >>= 1){
		if ((threadIdx.x < i) && (cache_d_Dist[threadIdx.x] > cache_d_Dist[threadIdx.x + i])){
			cache_d_Dist [threadIdx.x] = cache_d_Dist [threadIdx.x + i];
			cache_d_Pairs[threadIdx.x] = cache_d_Pairs[threadIdx.x + i];
		}
		__syncthreads();
	}
	// We now have the min value stored in cache[0] and 
	__syncthreads();
	//write back to each element in d_Dist[] and d_Pairs[] the min value - pair
	if (threadIdx.x == 0){
		d_Dist [0] = cache_d_Dist [0];
		d_Pairs[0] = cache_d_Pairs[0];
	}
}


int main( void ) {
	unsigned int i, j; 
	unsigned long long count = 0, *dev_count;
	float elapsedTime, min_dist = FLT_MAX, *dev_min_dist, distance, *d_Dist;
	float2 *P, *dev_P;
	float4 *d_Pairs, min_pair;

	//get device properties
	cudaDeviceProp prop;
	cudaGetDeviceProperties( &prop, 0 );
	
	//timers for measuring performance
	cudaEvent_t start, stop;
	cudaEventCreate( &start );
	cudaEventCreate( &stop );
	
	// Allocate input array P
	P = (float2*)malloc( N*sizeof(float2) );
	
	//initialize P array in host from file
    ifstream file_reader_x( "E:\\rand_x_num.txt" );
	if ( ! file_reader_x.is_open())		printf("Could not open rand_x_num file!\n");
	
	ifstream file_reader_y( "E:\\rand_y_num.txt" );
	if ( ! file_reader_y.is_open())		printf("Could not open rand_y_num file!\n");

	for ( int i = 0; i < N; i++ ){
		float point;
		file_reader_x >> point;		P[i].x = point;
		file_reader_y >> point;		P[i].y = point;
	}
	// initialize the array P on the CPU
	/*for(i=0;i<N;i++){
		P[i].x = ((float) (N)/ (float)i );
		P[i].y = ((float) (N)/ (float)i );
		//P[i].x = ((float)rand() / ((float)RAND_MAX));
		//P[i].y = ((float)rand() / ((float)RAND_MAX));
	}*/

	printf("\t\t--- N = %d \t\t---\n\t\t--- Blocks = %d \t---\n\t\t--- Threads = %d \t---\n\n", N, BLOCKS, BLOCK_SIZE);
	
	//CPU verification results.
	if (N < 65537){
		for (i=0; i<(N-1); i++){
			for (j=i+1; j<N; j++){
				count++;
				//calculate distance of current float2s
				distance = (P[i].x - P[j].x) *(P[i].x - P[j].x) + 
					(P[i].y - P[j].y) *(P[i].y - P[j].y);
				if (distance < min_dist)
					min_dist = distance;
			}
		}
		printf("\t\t---- CPU ----\n--- Min_dist = %.24f ---\n--- counts = %ul---\n\n", sqrt(min_dist), count);
		min_dist = FLT_MAX;
		count = 0;
	}

	// allocate the memory on the GPU
	cudaMalloc( (void**) &dev_P,				 N * sizeof(float2				));
	cudaMalloc( (void**) &dev_min_dist,				 sizeof(float				));
	cudaMalloc( (void**) &dev_count,				 sizeof(unsigned long long	));
	cudaMalloc( (void**) &d_Dist,	(N/BLOCK_SIZE) * sizeof(float				));
	cudaMalloc( (void**) &d_Pairs,	(N/BLOCK_SIZE) * sizeof(float4				));
	
	// copy the array P and min_distance to the GPU
	cudaMemcpy( dev_P,			P,			N * sizeof(float2),				cudaMemcpyHostToDevice);
	cudaMemcpy( dev_min_dist,	&min_dist,		sizeof(float),				cudaMemcpyHostToDevice);
	cudaMemcpy( dev_count,		&count,			sizeof(unsigned long long), cudaMemcpyHostToDevice);
	//printf("\nAfter cudamalloc and memcopy\nMin_dist = %g\n\n\n", min_dist);
	cudaDeviceSynchronize();
	// capture the start time
	cudaEventRecord( start, 0 ) ;
	

	//Launch Kernel with maximum blocks defined by prop.maxGridSize[0] or BLOCKS
	//cudaFuncSetCacheConfig(compare_float2s_BF, cudaFuncCachePreferL1);
	compare_float2s_BF	<<<	BLOCKS < prop.maxGridSize[0] ? BLOCKS : prop.maxGridSize[0], BLOCK_SIZE >>> ( dev_count, dev_P, d_Dist, d_Pairs );
	
	//cudaFuncSetCacheConfig(MinDistReduction, cudaFuncCachePreferL1);
	MinDistReduction	<<<	1, BLOCK_SIZE >>> ( d_Dist, d_Pairs );
	
	cudaDeviceSynchronize();
	// get stop time, and display the timing results
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );


	// copy results back from the GPU to the CPU
	//cudaMemcpy( P, dev_P, sizeof(float2), cudaMemcpyDeviceToHost );
	cudaMemcpy( &min_dist, d_Dist, sizeof(float),				cudaMemcpyDeviceToHost );
	cudaMemcpy( &min_pair, d_Pairs, sizeof(float4),				cudaMemcpyDeviceToHost );
	cudaMemcpy( &count, dev_count, sizeof(unsigned long long),	cudaMemcpyDeviceToHost );


	cudaEventElapsedTime( &elapsedTime, start, stop );


	cudaEventDestroy( start );
	cudaEventDestroy( stop );
	printf("\t\t---- GPU ----\n--- Min_dist = %.24f ---\n", sqrt(min_dist) );
	printf("point = (%.12f, %.12f) \n", min_pair.x, min_pair.y );
	printf("point = (%.12f, %.12f) \n", min_pair.z, min_pair.w );
	printf( "--- Time to generate: %.9f sec ---\n", elapsedTime/1000 );
	printf("--- Counts = %d ---\n", count );
	
	// free the memory allocated on the GPU
	cudaFree( dev_P );
	cudaFree( dev_min_dist );
	cudaFree( dev_count );
	cudaFree( d_Dist );
	cudaFree( d_Pairs );
	free(P);

	cudaDeviceReset(); //for visual profiler
	getchar();
	return 0;
}