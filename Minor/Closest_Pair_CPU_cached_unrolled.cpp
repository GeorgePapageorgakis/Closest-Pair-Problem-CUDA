#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <float.h>

typedef struct { float x, y; } point;

void print_array(int N,point *P){
	int i;
	for (i=0;i<N;i++){
		printf("P[%d] = (%.12f,%.12f)\n",i,P[i].x,P[i].y);fflush(stdout);
	}
}


float compare_points_BF(register int N, register int B, point *P){
	register int i, j, ib, jb, iin, jjn, num_blocks = (N + (B-1)) / B;
	point *p1, *p2;
	register float distance=0, min_dist=FLT_MAX, regx, regy;
	//unsigned long long calc = 0;

	//break array data in N/B blocks
	for (i = 0; i < num_blocks; i++){
		for (j = i; j < num_blocks; j++){
			if ((j+1)*B < N){
				jjn = (j+1)*B;
				//reads the moving frame block to compare with the i block
				for (jb = j * B; jb < jjn; jb++){
					//avoid float comparisons that occur when i block = j block
					//Register Allocated
					regx = P[jb].x;
					regy = P[jb].y;
					if ((i+1)*B < N){
						iin = (i+1)*B;
						for (i == j ? (ib = jb + 1) : (ib = i * B); ib < iin; ib++){
							if((distance = (P[ib].x - regx) * (P[ib].x - regx) + (P[ib].y - regy) * (P[ib].y - regy)) < min_dist){
								min_dist = distance;
								p1 = &P[ib];
								p2 = &P[jb];
							}
						}
					}
					else{
						for (i == j ? (ib = jb + 1) : (ib = i * B); ib < N; ib++){
							if((distance = (P[ib].x - regx) * (P[ib].x - regx) + (P[ib].y - regy) * (P[ib].y - regy)) < min_dist){
								min_dist = distance;
								p1 = &P[ib];
								p2 = &P[jb];
							}
						}
					}
				}
			}
			else{
				for (jb = j * B; jb < N; jb++){
					regx = P[jb].x;
					regy = P[jb].y;
					if ((i+1)*B < N){
						iin = (i+1)*B;
						for (i == j ? (ib = jb + 1) : (ib = i * B); ib < iin; ib++){
							if((distance = (P[ib].x - regx) * (P[ib].x - regx) + (P[ib].y - regy) * (P[ib].y - regy)) < min_dist){
								min_dist = distance;
								p1 = &P[ib];
								p2 = &P[jb];
							}
						}
					}
					else{
						for (i == j ? (ib = jb + 1) : (ib = i * B); ib < N; ib++){
							if((distance = (P[ib].x - regx) * (P[ib].x - regx) + (P[ib].y - regy) * (P[ib].y - regy)) < min_dist){
								min_dist = distance;
								p1 = &P[ib];
								p2 = &P[jb];
							}
						}

					}
				}
			}
		}
	}
	//printf("%lu calculations\t", calc);
	//printf("\nMin_Dist = %.12f\nP[%d] = (%.12f,%.12f)--P[%d] = (%.12f,%.12f)\n",
	//		sqrt(min_dist),p1,P[p1].x,P[p1].y,p2,P[p2].x,P[p2].y);fflush(stdout);
	return sqrt(min_dist);
}

int main (void){
	time_t start; srand((float)time(NULL));
	float timeTaken, min_dist;
	int cs, pe, i, blk_fctr = 128, N = 8192;

	//itterate with incremental # of N points and block size for Point Elements and Cache Size
	for (pe=0;pe<10;pe++){
		for(cs=0;cs<13;cs++){
			//create 1D array of 2 x #points (x,y)
			point *points = (point*) malloc (sizeof(*points) * N);
			if(points==NULL){
				free(points);
				printf("ERROR Allocating Memory for points[]\n");
				exit(1);
			}

			//i for every point(x, y)
			for(i=0;i<N;i++){
				//initialize array of points
				//points[i].x = ((float)rand() / ((float)RAND_MAX)*1000);
				//points[i].y = ((float)rand() / ((float)RAND_MAX)*1000);
				points[i].x = ((float) (i)/ (float)N );
				points[i].y = ((float) (i)/ (float)N );
			}

			//print_array(N,points);

			/* Start timer */
			start = clock();

			min_dist = compare_points_BF(N, blk_fctr, points);

			/* Stop timer */
			timeTaken = (float)(clock() - start) / CLOCKS_PER_SEC;
			printf("Min_dist = %f\tBlock_size = %d\tN = %d\tRun time: %.3f sec\n",
					min_dist, blk_fctr, N, timeTaken);

			blk_fctr *= 2;
			free(points);
		}
		N *= 2;
		blk_fctr = 128;
		printf("\n");
	}
	system("PAUSE");
	return 0;
}
