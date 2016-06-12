/*
 * The closest pair of points problem or closest pair problem is a problem of computational geometry:
 * given n points in metric space, find a pair of points with the smallest distance between them. 
 * The closest pair problem for points in the Euclidean plane[1] was among the first geometric problems
 * which were treated at the origins of the systematic study of the computational complexity of geometric algorithms.
 * The problem may be solved in O(n log n) time in a Euclidean space or Lp space of fixed dimension d.
 * In the algebraic decision tree model of computation, the O(n log n) algorithm is optimal. The optimality
 * follows from the observation that the element uniqueness problem is reducible to the closest pair problem:
 * checking whether the minimal distance is 0 after the solving of the closest pair problem answers the 
 * question whether there are two coinciding points. In the computational model which assumes that the floor 
 * function is computable in constant time the problem can be solved in O(n log log n) time. If we allow
 * randomization to be used together with the floor function, the problem can be solved in O(n) time.
 *
 * Default caching mode: load granularity 128 bytes, attemps hit in L1->L2->GMEM
 * -Xptxas -dlcm=ca
 * non-caching mode: load granularity 32 bytes, attemps hit in L2->GMEM
 * -Xptxas -dlcm=cg 
 * Compiled for CC 3.0 devices
 * 
 * Author: George Papageorgakis, 2013
 */

// CUDA Runtime
#include <cuda_runtime.h>
// Utilities and system includes
#include <cuda.h>
#include "helper_cuda.h"
#include "helper_timer.h"
#include "sortingNetworks_common.h"

using namespace std;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Test driver
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv){
    cudaError_t error;
    printf("%s Starting...\n\n", argv[0]);
	printf("Starting up CUDA context...\n");
    int dev = findCudaDevice(argc, (const char **)argv);

	bool	*h_S_yy, *d_S_yy;
	float	*h_Dist, *d_Dist;
	float2	*h_P, *h_S_x, *h_S_y, *d_S_x, *d_S_y;
	float4	*h_Pairs, *d_Pairs;

    StopWatchInterface *hTimer = NULL;
    const uint			DIR	   = 1;  //direction 1 ascending, 0 descending

    printf("Allocating and initializing host arrays...\n\n");
    sdkCreateTimer(&hTimer);

    h_P		= (float2 *)malloc(	N		* sizeof(float2));
    h_S_x	= (float2 *)malloc(	N		* sizeof(float2));
	h_S_y	= (float2 *)malloc(	N		* sizeof(float2));
	h_S_yy	= (bool	  *)malloc(	N		* sizeof(bool  ));
	h_Dist	= (float  *)malloc((N/base)	* sizeof(float ));
	h_Pairs	= (float4 *)malloc((N/base) * sizeof(float4));


	if ( argc != 3 ){
		cout << "Initializing host points from default files...\n\n";
		//initialize P array in host from file
		ifstream file_reader_x( "E:\\v2_x_num.txt" );
		if ( ! file_reader_x.is_open()) { cout << "Could not open rand_x_num file!\n"; }
		ifstream file_reader_y( "E:\\v2_y_num.txt" );
		if ( ! file_reader_y.is_open())	{ cout << "Could not open rand_y_num file!\n"; }

		if ( ! file_reader_x.is_open() && ! file_reader_y.is_open() ){
			cout << "Initializing points with values...\n\n";
			for (uint i = 0; i < N; i++){
				h_S_yy[i] = false;
			}
			// 0 for uniform, 1 for normal distributions
			distribution_init(h_P, 0);
		}
		else{
			for (uint i = 0; i < N; i++){
				float point;
				h_S_yy[i] = false;
				file_reader_x >> point;		h_P[i].x = point;
				file_reader_y >> point;		h_P[i].y = point;
			}
		}
	}
	else{
		cout << "Initializing host points from custom files...\n\n";
		ifstream file_reader_x( argv[1] );
		if ( ! file_reader_x.is_open()){ cout << "Could not open file, " << argv[ 1 ] << endl; }
		ifstream file_reader_y( argv[2] );
		if ( ! file_reader_y.is_open()){ cout << "Could not open file " << argv[ 2 ] << endl; }
		cout << "Need an absolute path of the file" << endl;
		if ( file_reader_x.is_open() && file_reader_y.is_open() ){
			for (uint i = 0; i < N; i++){
				float point;
				h_S_yy[i] = false;
				file_reader_x >> point;		h_P[i].x = point;
				file_reader_y >> point;		h_P[i].y = point;
			}
		}
	}

    printf("Allocating and initializing CUDA arrays...\n\n");
    error = cudaMalloc((void**) &d_S_yy,	 (N/base) * sizeof(bool  ));	checkCudaErrors(error);
	error = cudaMalloc((void**) &d_S_x,				N * sizeof(float2));	checkCudaErrors(error);
	error = cudaMalloc((void**) &d_S_y,				N * sizeof(float2));	checkCudaErrors(error);
	error = cudaMalloc((void**) &d_Dist,	 (N/base) * sizeof(float ));	checkCudaErrors(error);
	error = cudaMalloc((void**) &d_Pairs,	 (N/base) * sizeof(float4));	checkCudaErrors(error);

	//copy initialized values to d_S_y for device memory-save
	error = cudaMemcpy(d_S_y, h_P, N * sizeof(float2), cudaMemcpyHostToDevice);   checkCudaErrors(error);
   	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Closest Pair
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	error = cudaDeviceSynchronize();	checkCudaErrors(error);
	sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);

	//INITIAL BITONIC SORT of P (or d_S_y) into S_x as of x
	bitonicSort( d_S_x, d_S_y, 1, N, DIR, 0 );

	//ONE-TIME Run of Brute force for each batch of size base
	bruteForce ( d_S_x, d_Dist, d_Pairs);

	uint batchLength;
	for (batchLength = 2*base; batchLength <= N; batchLength <<= 1){
		//# of subarrays
		uint batchsize =  N / batchLength;
			
		//Sort of S_y array based on batch properties
		//oddEvenMergeSort( d_S_y, d_S_x, batchsize, batchLength, DIR, 1 );
		bitonicSort( d_S_y, d_S_x, batchsize, batchLength, DIR, 1 );

		closest_pair (d_S_yy, d_S_x, d_S_y, d_Dist, d_Pairs, batchLength);		
	}
	error = cudaDeviceSynchronize();				checkCudaErrors(error);
	sdkStopTimer(&hTimer);
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	error = cudaMemcpy(h_Dist,	d_Dist,	 (N/base) * sizeof(float ), cudaMemcpyDeviceToHost);    checkCudaErrors(error);
	error = cudaMemcpy(h_Pairs,	d_Pairs, (N/base) * sizeof(float4), cudaMemcpyDeviceToHost);    checkCudaErrors(error);

	printf("Closest Pair point:\nCP_dist = %.20f\n[ %.18f, %.18f ]\n[ %.18f, %.18f ]\n ", h_Dist[0], h_Pairs[0].x, h_Pairs[0].y, h_Pairs[0].z, h_Pairs[0].w);

    printf("\n --- Average time: %f ms ---\n\n", sdkGetTimerValue(&hTimer));
	
    double dTimeSecs = 1.0e-3 * sdkGetTimerValue(&hTimer);
    printf("Sorting Networks-bitonic\nThroughput = %.4f MElements/s, Time = %.5f s, Size = %u elements\nNumSteps = %u\nNumDevsUsed = %u\n",
			(1.0e-6 * (double)batchLength/dTimeSecs), dTimeSecs, N, 1, 1);
			
    printf("Shutting down...\nPress any key to terminate...\n");
    sdkDeleteTimer(&hTimer);

	//free memory
	cudaFree(d_S_x);
	cudaFree(d_S_y);
	cudaFree(d_S_yy);
	cudaFree(d_Dist);
	cudaFree(d_Pairs);
    free(h_P);
	free(h_S_x);
	free(h_S_y);
	free(h_S_yy);
	free(h_Dist);
	free(h_Pairs);
	// REMOVE getchar for Visual Profiler analysis
	getchar();
	//reset for NSight sync
    cudaDeviceReset();
    //exit(flag ? EXIT_SUCCESS : EXIT_FAILURE);
}
