#define _USE_MATH_DEFINES // for C++
#include <cmath>
#define _USE_MATH_DEFINES // for C
#include <math.h>

#include <iostream>
#include <time.h>
#include <algorithm>
#include <vector_types.h>
#include <random>
#include "sortingNetworks_common.h"


unsigned time_seed()
{
	time_t now = time ( 0 );
	unsigned char *p = (unsigned char *)&now;
	unsigned seed = 0;
	size_t i;
 
	for ( i = 0; i < sizeof now; i++ )
		seed = seed * ( UCHAR_MAX + 2U ) + p[i];
 
	return seed;
}

float randn (float mu, float sigma)
{
	float U1, U2, W, mult;
	static float X1, X2;
	static int call = 0;
 
	if (call == 1) {
		call = !call;
		return (mu + sigma * (float) X2);
	}
 
	do {
		U1 = -1 + ((float) rand () / (float) RAND_MAX) ;
		U2 = -1 + ((float) rand () / (float) RAND_MAX) ;
		W = pow (U1, 2) + pow (U2, 2);
	}while (W >=1 || W == 0);
 
	mult = sqrt ((-2 * log (W)) / W);
	X1 = U1 * mult;
	X2 = U2 * mult;
 
	call = !call;
 
	return (mu + sigma * (float) X1);
}

//initialize uniform (0) or normal (1) distribution points
void distribution_init(float2 *P, uint dist_type)
{
	std::srand((float)time_seed());
	float * arr = new float[N];
	
	if (dist_type == 0){
		for(uint i=0; i<N; i++)
			arr[i] = (float(float(i) / float(N)));
		arr[0] = float(float(0.9) / float(N));

		std::random_shuffle(arr, arr + N);    
		for ( uint i = 0; i < N; i++ )
			P[i].x = arr[i];
	
		std::random_shuffle(arr, arr + N);    
		for ( uint i = 0; i < N; i++ )
			P[i].y = arr[i];	
	}
	else{
		for(int i=0; i<N; i++)
			arr[i] = randn(0, 0.0000001);

		std::random_shuffle(arr, arr + N);    
		for ( uint i = 0; i < N; i++ )
			P[i].x = arr[i];

		std::random_shuffle(arr, arr + N);    
		for ( uint i = 0; i < N; i++ )
			P[i].y = arr[i];
	}
	free(arr);
}
