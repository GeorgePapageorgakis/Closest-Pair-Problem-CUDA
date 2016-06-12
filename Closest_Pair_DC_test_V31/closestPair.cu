//#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_functions.h>
#include "helper_cuda.h"
#include "sortingNetworks_common.h"
#include "sortingNetworks_common.cuh"

__device__ __constant__ float	CONST_D_DIST;
__device__ __constant__ float4  CONST_D_PAIR;

//Calculate distance of each block of size = base.
__global__ void brute_Force(
	float2	*d_S_x,
	float	*d_Dist,
	float4	*d_Pairs
)
{
	uint	tid	   = threadIdx.x + blockIdx.x * blockDim.x;
	uint	stride = blockDim.x * gridDim.x;
	float	local_dist, local_min_dist;
	float4	local_ClosestPair;

	__shared__ float2 cache_S_x		[base];
	__shared__ float  cache_d_Dist	[base];
	__shared__ float4 cache_d_Pairs	[base];

	while (tid < N){
		//fetch data from GLOBAL memory to SHARED memory, coalesced memory access
		cache_S_x[threadIdx.x] = d_S_x[tid];
		__syncthreads();
		local_min_dist = FLT_MAX;

		//Each thread compares its onwn point with the onward rest points in the block
		//get the beginning of the cached block, check boundaries of index
		for (uint i = threadIdx.x + 1; i < blockDim.x; i++){
			local_dist = (cache_S_x[i].x - cache_S_x[threadIdx.x].x) * (cache_S_x[i].x - cache_S_x[threadIdx.x].x) + (cache_S_x[i].y - cache_S_x[threadIdx.x].y) * (cache_S_x[i].y - cache_S_x[threadIdx.x].y);
			if (local_dist < local_min_dist){
				local_min_dist		= local_dist;
				local_ClosestPair.x = cache_S_x[threadIdx.x].x;
				local_ClosestPair.y = cache_S_x[threadIdx.x].y;
				local_ClosestPair.z = cache_S_x[i].x;
				local_ClosestPair.w = cache_S_x[i].y;
			}
		}
		cache_d_Dist [threadIdx.x] = local_min_dist;
		cache_d_Pairs[threadIdx.x] = local_ClosestPair;
		// Synchronize so the preceding computation is done before loading new elements
		__syncthreads();

		// Parallel Reduction: Sequential Addressing in shared mem (bitwise right shift i) 
		// reversed loop and threadID-based indexing
		for (int i = blockDim.x / 2; i > 0; i >>= 1){
			if ((threadIdx.x < i) && (cache_d_Dist[threadIdx.x] > cache_d_Dist[threadIdx.x + i])){
				cache_d_Dist [threadIdx.x] = cache_d_Dist [threadIdx.x + i];
				cache_d_Pairs[threadIdx.x] = cache_d_Pairs[threadIdx.x + i];
			}
			__syncthreads();
		}
		// We now have the min value of each block stored in cache_CP[0] and 
		// we can store it in the corresponding dev_Closest_Pair[] for later global memory reduction;
		if (threadIdx.x == 0){
			d_Dist	[tid/base] = sqrtf(cache_d_Dist [0]);
			d_Pairs	[tid/base] = cache_d_Pairs[0];
		}
		tid += stride;
		__syncthreads();
	}
}


//Find minimum distance and pair. Parallel Reduction: Sequential Addressing
__global__ void minDistReduction(
	float   *d_Dist,
	float4  *d_Pairs
)
{
	uint tid = threadIdx.x + blockIdx.x * blockDim.x;
	uint stride = blockDim.x * gridDim.x;
	float local_d_Dist;
	float4 local_d_Pairs;

	__shared__ float  cache_d_Dist [1024];
	__shared__ float4 cache_d_Pairs[1024];

	//fetch data from GLOBAL memory, coalesced memory access
	local_d_Dist  = d_Dist [tid];
	local_d_Pairs = d_Pairs[tid];
	__syncthreads();
	tid += stride;

	while (tid < N/base){
		//Each thread compares its local element with only one +stride element in d_Dist[].
		if (local_d_Dist > d_Dist[tid] ){
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
	for(uint i = blockDim.x / 2; i > 0; i >>= 1){
		if ((threadIdx.x < i) && (cache_d_Dist[threadIdx.x] > cache_d_Dist[threadIdx.x + i])){
			cache_d_Dist [threadIdx.x] = cache_d_Dist [threadIdx.x + i];
			cache_d_Pairs[threadIdx.x] = cache_d_Pairs[threadIdx.x + i];
		}
		__syncthreads();
	}
	//We now have the min value stored in d_Dist[0] and d_Pairs[0]
	//write back in d_Dist[] and d_Pairs[] the min value - pair
	if (threadIdx.x == 0){
		d_Dist [0] = cache_d_Dist [0];
		d_Pairs[0] = cache_d_Pairs[0];
	}
}


__global__ void findVerticalStrip(
	bool	*d_S_yy,
	float2	*d_S_x,
	float2	*d_S_y,
	uint	batchLength
)
{
	uint tid	= threadIdx.x + blockIdx.x * blockDim.x;
	uint stride = blockDim.x * gridDim.x;
	float local_dist;
	float Lx;
	__shared__ bool  block_has_strip_point;

	while (tid < N) {
		if (threadIdx.x == 0)
			block_has_strip_point = false;
		__syncthreads();
		//Get the corresponding middle point L from S_x[]
		Lx = d_S_x[(tid / batchLength) * batchLength + (batchLength / 2)].x;
		
		//compare each element of S_y with point L. L point is always in S_yy
		//if TRUE then the block has a point in vertical stip.
		local_dist = d_S_y[tid].x - Lx;
		// |local_dist| < loc_min_dist 
		if ( (local_dist <= CONST_D_DIST) && (local_dist >= (-CONST_D_DIST)) ){
			if ( block_has_strip_point == false)
				block_has_strip_point = true;
		}
		__syncthreads();

		if (threadIdx.x == 0 && block_has_strip_point != false){
			d_S_yy[tid/base] = true;
		}
		tid += stride;
		__syncthreads();
	}
}


__global__ void compareStripPoints(
	bool	*d_S_yy,
	float2	*d_S_x,
	float2	*d_S_y,
	float   *d_Dist,
	float4  *d_Pairs,
	uint	batchLength
)
{
	uint	tid	 = threadIdx.x + blockIdx.x * blockDim.x;
	uint	stride = blockDim.x * gridDim.x;
	bool	tid_is_strip_point;
	float	Lx,	loc_min_dist;
	float4	loc_min_pair;

	__shared__ int		block_count;
	__shared__ float2	cache_d_S_y		[base];
	__shared__ float	cache_d_Dist	[base];
	__shared__ float4	cache_d_Pairs	[base];
	
	//copy from shared to local regs, all threads/block have the same min_dist
	loc_min_dist = CONST_D_DIST;
	loc_min_pair = CONST_D_PAIR;
	while (tid < N){
		//compute only if there are strip points in the current block points
		if (d_S_yy [tid/base] != false){
			uint	sub_tid		 = tid; 
			//uint	loc_sp_count = 0;
			float	loc_dist;
			float2	local_strip_point;

			tid_is_strip_point = false;
			block_count = 0;
			cache_d_S_y	[threadIdx.x] = d_S_y [tid];
			//__syncthreads();
			
			//load to local reg the corresponding points to compare on this thread
			local_strip_point = cache_d_S_y[threadIdx.x];
			//Get the corresponding middle point L from S_x[]
			Lx = d_S_x[(tid / batchLength) * batchLength + (batchLength / 2)].x;
			
			//check if point[tid] is within vertical strip
			loc_dist = local_strip_point.x - Lx; 
			tid_is_strip_point = ( (loc_dist <= CONST_D_DIST) && (loc_dist >= (-CONST_D_DIST)) );

			//compare with the onward blocks, get the limits of the current batch
			while ((sub_tid  < (tid / batchLength) * batchLength + batchLength) && block_count < 7){
				if (d_S_yy[sub_tid/base] != false){
					//avoid reloading
					if (sub_tid != tid){
						if (threadIdx.x == 0)
							block_count ++;
						cache_d_S_y [threadIdx.x] = d_S_y [sub_tid ];
					}
					__syncthreads();
					//compare only if corresponding point of threadIdx.x was in the strip
					if (tid_is_strip_point != false){
						//Compare each point in the strip, diff index if in the initial block or in the following ones
						for (uint i = (tid == sub_tid ? threadIdx.x + 1 :  0); i < blockDim.x; i++){
							//check if point[sub_tid] is within vertical strip
							loc_dist = cache_d_S_y[i].x - Lx;				
							if ( (loc_dist <= CONST_D_DIST) && (loc_dist >= (-CONST_D_DIST)) ){
								//loc_sp_count++;
								loc_dist = sqrtf((cache_d_S_y[i].x - local_strip_point.x) * (cache_d_S_y[i].x - local_strip_point.x)
												+ (cache_d_S_y[i].y - local_strip_point.y) * (cache_d_S_y[i].y - local_strip_point.y));
								if (loc_dist < loc_min_dist){
									loc_min_dist	= loc_dist;
									loc_min_pair.x	= local_strip_point.x;
									loc_min_pair.y	= local_strip_point.y;
									loc_min_pair.z	= cache_d_S_y[i].x;
									loc_min_pair.w	= cache_d_S_y[i].y;
								}
							}
						}
					}
				}
				sub_tid += blockDim.x;
				__syncthreads();
			}
			cache_d_Dist [threadIdx.x] = loc_min_dist;
			cache_d_Pairs[threadIdx.x] = loc_min_pair;
			// Synchronize so that the preceding computation is done before loading new elements in next iteration
			__syncthreads();
			// Parallel Reduction: Sequential Addressing in shared mem (bitwise right shift i) 
			// reversed loop and threadID-based indexing
			for(int i = blockDim.x / 2; i > 0; i >>= 1){
				if ((threadIdx.x < i) && (cache_d_Dist[threadIdx.x] > cache_d_Dist[threadIdx.x + i])){
					cache_d_Dist [threadIdx.x] = cache_d_Dist [threadIdx.x + i];
					cache_d_Pairs[threadIdx.x] = cache_d_Pairs[threadIdx.x + i];
				}
				__syncthreads();
			}
			// We now have the min value of each block stored in cache_d_Dist[0] and cache_d_Pairs[0]
			// we can store it in the corresponding d_Dist[] and d_Pairs[] for future reduction;
			if (threadIdx.x == 0){
				d_Dist	[tid/base] = cache_d_Dist [0];
				d_Pairs [tid/base] = cache_d_Pairs[0];
			}
		}
		tid += stride;
		__syncthreads();		
	}
}


////////////////////////////////////////////////////////////////////////////////
// Interface function
////////////////////////////////////////////////////////////////////////////////
/* Pass the already sorted arrays S_x and S_y.
 * Array S_y is already merged (2 consecutive subarrays of S_x) and
 * sorted as of x from previous step. The only job here is to calculate
 * point L, array S_yy and distance between points within vertical strip S_yy.
*/
void closest_pair(
	bool	*d_S_yy,
	float2	*d_S_x,
	float2	*d_S_y,	
	float	*d_Dist,
	float4	*d_Pairs,   
    uint	batchLength //length of each batch
)
{
	cudaError_t error;
	cudaDeviceProp prop;
	cudaGetDeviceProperties( &prop, 0 );

	//Input the arrays merged by 2 subarrays of S_y and S_x with already d base
	//reduce array d_Dist[], d_Pairs[] to get min dist for next step
	minDistReduction <<< 1, prop.maxThreadsPerBlock >>>(d_Dist, d_Pairs);
	error =  cudaMemcpyToSymbolAsync( CONST_D_DIST, d_Dist,  sizeof(float ), 0, cudaMemcpyDeviceToDevice );	checkCudaErrors(error);
	error =  cudaMemcpyToSymbolAsync( CONST_D_PAIR, d_Pairs, sizeof(float4), 0, cudaMemcpyDeviceToDevice );	checkCudaErrors(error);
	
	//Global Sync.

	//reset all bool elements of the Vertical Strip array to 0
	error = cudaMemsetAsync(d_S_yy, false,  sizeof(bool) * (N/base));	checkCudaErrors(error);
	
	/* Create an array S_yy which is S_y with all points not in the 2d-wide 
	 * vertical strip removed. The array S_yy is sorted by y cordinates.
	 * For each point p in the array S_yy try to find the points in S_yy that are within distance d of p.
	 * Only the 7 next blocks-points in S_yy need to be considered.*/
	findVerticalStrip <<< ((N / base) < prop.maxGridSize[0]) ? (N / base) : prop.maxGridSize[0], base >>>(d_S_yy, d_S_x, d_S_y, batchLength);
	
	//Global Sync.
	
	/* Compute the distance from p to each of these 7 points and keep track of the 
	 * closest-pair distance d' found over all pairs of points in S_yy */
	compareStripPoints <<< ((N / base) < prop.maxGridSize[0]) ? (N / base) : prop.maxGridSize[0], base >>>(d_S_yy, d_S_x, d_S_y, d_Dist, d_Pairs, batchLength);
	
	if (batchLength == N)
		minDistReduction <<< 1, prop.maxThreadsPerBlock >>>(d_Dist, d_Pairs);
}

//base calculation of closest pairs coresponding to d_S_x index
void bruteForce(
	float2	*d_S_x,
	float	*d_Dist,
	float4	*d_Pairs
)
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties( &prop, 0 );
	brute_Force <<< (N / base) < prop.maxGridSize[0] ? (N / base) : prop.maxGridSize[0], base >>>(d_S_x, d_Dist, d_Pairs);
}