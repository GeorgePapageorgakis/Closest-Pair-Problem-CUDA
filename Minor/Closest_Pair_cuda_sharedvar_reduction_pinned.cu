#include "stdio.h"
#include "iostream"
#include "stdlib.h"
#include "time.h"
#include "math.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "cuComplex.h"
#include "device_functions.h"
#define N 65536/8
#define BLOCK_SIZE 1024
#define BLOCKS (N + (BLOCK_SIZE - 1))/BLOCK_SIZE


typedef struct { float x, y; } point;

//Constant Memory on GPU
__constant__ point dev_P[N];

float inf(void) {
  return (float)HUGE_VAL;
}
__device__ inline float atomicMin(float *addr, float value){
	float old = *addr, assumed;
	if( old <= value ) return old;
	do{
		assumed = old;
		old = atomicCAS((unsigned int*)addr, __float_as_int(assumed), __float_as_int(value));
	}while( old!=assumed );
	return old;
}


__global__ void compare_points_BF( unsigned long long *dev_count, float *gbl_min_dist) {

	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int stride = blockDim.x * gridDim.x;
	unsigned int cache_index, current_block, grid_offset = 0;
	float local_dist, local_min = * gbl_min_dist;
	point temp;
	__shared__ point cache[BLOCK_SIZE];
	__syncthreads();

	while (tid < N){
		//Each thread compares its onwn point with the onward rest points.
		temp.x = dev_P[tid].x;
		temp.y = dev_P[tid].y;

		//cached blocks of P array stored in shared memory as cache[], current_block corresponds to P's cached sub-array
		for ( current_block = blockIdx.x + (grid_offset * gridDim.x); current_block < BLOCKS; current_block++){
			//check for boundaries violation within each current block
			if (current_block * blockDim.x + threadIdx.x < N){
				//fetch data from GLOBAL memory to SHARED memory, coalesced memory access
				cache[threadIdx.x] = dev_P[current_block * blockDim.x + threadIdx.x];
			}
			//synchronize threads in this block
			__syncthreads();
			
			//get the beginning of the cached block or the next point if it is a comparison on the same block, check boundaries of index
			for(current_block == blockIdx.x + (grid_offset * gridDim.x) ? cache_index = threadIdx.x + 1 : cache_index=0; 
					(cache_index < blockDim.x) && (current_block * blockDim.x + cache_index) < N; 
					cache_index++){
				//atomicAdd( dev_count, 1); //Enable for verification (affects performance)
				//calculate distance of current points
				local_dist = (cache[cache_index].x - temp.x) * (cache[cache_index].x - temp.x) + (cache[cache_index].y - temp.y) * (cache[cache_index].y - temp.y);
				//Determine the minimum numeric value of the arguments
				if (local_dist < local_min)
					local_min = local_dist;
			}
			// Synchronize to make sure that the preceding computation
            // is done before loading new elements in the next iteration
			__syncthreads();
		}
		//for Arbitrarily P[] Length
		tid += stride;
		grid_offset ++;
		__syncthreads();
	}
	/*  Use atomic operations on both the local and global level. 
	*   Atomic operations are performed on shared memory, no access to global memory is required. 
	*	This means that atomic operations performed in shared memory are generally much faster than those
	*	performed on global memory. For CUDA devices with compute capability of 1.2 or above.
	*/
	__syncthreads();
	cache[threadIdx.x].x = local_min;
	// Do reduction in shared mem (bitwise right shift i)
    for(unsigned int i=blockDim.x/2; i>0; i>>=1){
        if (threadIdx.x < i)
			atomicMin(&(cache[threadIdx.x].x), cache[threadIdx.x + i].x);
        __syncthreads();
    }
	//__syncthreads();
	// We now have the min value stored in cache[0].x;
	if (threadIdx.x == 0)
		atomicMin( gbl_min_dist, cache[0].x );
}

int main( void ) {
	unsigned int i, j; 
	unsigned long long count = 0, *dev_count;
	float elapsedTime, min_dist = inf(), *dev_min_dist, distance;
	point *P;//, *dev_P;
	
	//get device properties
	cudaDeviceProp prop;
	cudaGetDeviceProperties( &prop, 0 );
	
	//timers for measuring performance
	cudaEvent_t start, stop;
	cudaEventCreate( &start );
	cudaEventCreate( &stop );
	
	// Allocate Pinned Host Memory input array P
	//cudaHostAlloc( (void**)&P, N * sizeof( *P ),cudaHostAllocDefault );
	
	P = (point*)malloc( N*sizeof(point) );
	
	// initialize the array P on the CPU
	for(i=0;i<N;i++){
		P[i].x = ((float) (N)/ (float)i );
		P[i].y = ((float) (N)/ (float)i );
		//P[i].x = ((float)rand() / ((float)RAND_MAX));
		//P[i].y = ((float)rand() / ((float)RAND_MAX));
	}

	printf("\t\t--- N = %d \t\t---\n\t\t--- Blocks = %d \t---\n\t\t--- Threads = %d \t---\n\n", N, BLOCKS, BLOCK_SIZE);
	
	//CPU verification results.
	if (N < 2*65537){
		for (i=0; i<(N-1); i++){
			for (j=i+1; j<N; j++){
				count++;
				//calculate distance of points
				distance = (P[i].x - P[j].x) *(P[i].x - P[j].x) + 
					(P[i].y - P[j].y) *(P[i].y - P[j].y);
				if (distance < min_dist)
					min_dist = distance;
			}
		}
		printf("\t\t---- CPU ----\n--- Min_dist = %.21f ---\n--- counts = %ul---\n\n", sqrt(min_dist), count);
		min_dist = inf();
		count = 0;
	}

	// allocate the memory on the GPU
	cudaMalloc( (void**) &dev_min_dist, sizeof(float) );
	cudaMalloc( (void**) &dev_count, sizeof(unsigned long long) );
	
	// copy the array P and min_distance to the GPU
	cudaMemcpyToSymbol( dev_P, P, sizeof(point) * N);
	cudaMemcpy( dev_min_dist, &min_dist, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy( dev_count, &count, sizeof(unsigned long long), cudaMemcpyHostToDevice);
	printf("\nAfter cudamalloc and memcopy\nMin_dist = %f\n\n\n", min_dist);
	
	// capture the start time
	cudaEventRecord( start, 0 ) ;
	

	//Launch Kernel with maximum blocks defined by prop.maxGridSize[0] or BLOCKS
	compare_points_BF <<< BLOCKS < prop.maxGridSize[0] ? BLOCKS : prop.maxGridSize[0], BLOCK_SIZE >>> ( dev_count, dev_min_dist );
	

	// copy results back from the GPU to the CPU
	cudaMemcpy( &min_dist, dev_min_dist, sizeof(float), cudaMemcpyDeviceToHost );
	cudaMemcpy( &count, dev_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost );

	// get stop time, and display the timing results
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );

	cudaEventElapsedTime( &elapsedTime, start, stop );


	cudaEventDestroy( start );
	cudaEventDestroy( stop );
	printf("\t\t---- GPU ----\n--- Min_dist = %.21f ---\n", sqrt(min_dist) );
	printf( "--- Time to generate: %.9f sec ---\n", elapsedTime/1000 );
	printf("--- Counts = %ul ---\n", count );
	
	// free the memory allocated on the GPU
	//cudaFree( dev_P );
	cudaFree( dev_min_dist );
	cudaFree( dev_count );
	//cudaFreeHost( P );
	free(P);

	cudaDeviceReset(); //for visual profiler
	getchar();
	return 0;
}