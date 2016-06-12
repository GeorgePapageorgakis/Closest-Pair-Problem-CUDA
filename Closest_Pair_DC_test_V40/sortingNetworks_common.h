#ifndef SORTINGNETWORKS_COMMON_H
#define SORTINGNETWORKS_COMMON_H

////////////////////////////////////////////////////////////////////////////////
// Shortcut definition
////////////////////////////////////////////////////////////////////////////////
typedef unsigned int	uint;

extern unsigned time_seed();

//extern float log_2(float N);

//N and base must be power of 2.
const uint             N = 1048576;
const uint          base = 128; //floor number of elements for brute force. power of 2

extern void distribution_init(
	float2	*P,
	uint dist_type
);

////////////////////////////////////////////////////////////////////////////////
// CUDA closest pair dc
////////////////////////////////////////////////////////////////////////////////
void closest_pair(
	uint	*d_S_yy,
	float2	*d_S_x,
	float2	*d_S_y,
	float	*d_Dist,
	float4	*d_Pairs,
    uint	batchLength
);

void bruteForce(
	float2	*d_S_x,
	float	*d_Dist,
	float4	*d_Pairs
);

////////////////////////////////////////////////////////////////////////////////
// CUDA sorting networks
////////////////////////////////////////////////////////////////////////////////
extern "C" uint bitonicSort(
    float2	*d_P_out,
    float2	*d_P_in,
    uint	batchSize,
    uint	arrayLength,
    uint	dir,
	uint	xy
);

extern "C" void oddEvenMergeSort(
    float2	*d_P_out,
    float2	*d_P_in,
    uint	batchSize,
    uint	arrayLength,
    uint	dir,
	uint	xy
);

#endif
