#ifndef SORTINGNETWORKS_COMMON_CUH
#define SORTINGNETWORKS_COMMON_CUH

#include "sortingNetworks_common.h"

//Enables maximum occupancy
#define SHARED_SIZE_LIMIT 1024U

//Map to single instructions on G8x / G9x / G100
#define UMUL(a, b)		__umul24((a), (b))
#define UMAD(a, b, c) ( UMUL((a), (b)) + (c) )

//comparator for sorting networks
__device__ inline void Comparator(
    float2 &keyA,
    float2 &keyB,
    uint dir,
	uint xy )
{
    float2 t;

	if (xy == 0){
		if ((keyA.x > keyB.x) == dir) {
			t = keyA;
			keyA = keyB;
			keyB = t;
		}
	}
	else{
		if ((keyA.y > keyB.y) == dir) {
			t = keyA;
			keyA = keyB;
			keyB = t;
		}
	}	
}

#endif
