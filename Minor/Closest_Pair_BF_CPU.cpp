#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <float.h>

typedef struct { float x, y; } point;


void print_array(int N, point *P){
	int i;
	for (i=0;i<N;i++){
		printf("P[%d] = (%.3f,%.3f)\n",i,P[i].x,P[i].y);fflush(stdout);
	}
}

float compare_points_BF(int N,point *P){
	  int i,j;
	  float  distance=0, min_dist=FLT_MAX;
	 point *p1, *p2;
	unsigned long long calc = 0;
	for (i=0;i<(N-1);i++){
		for (j=i+1;j<N;j++){

			//dx = P[i].x - P[j].x;
			//dy = P[i].y - P[j].y;

			//calculate distance of current points
			//distance = (dx*dx) + (dy*dy);
			//calc++;
			if ((distance = (P[i].x - P[j].x) * (P[i].x - P[j].x) +
					(P[i].y - P[j].y) * (P[i].y - P[j].y)) < min_dist){
				min_dist = distance;
				p1 = &P[i];
				p2 = &P[j];
			}
		}
	}
	//printf("%lu calculations\t", calc);
	//printf("p1 = %d, p2 = %d\t",p1, p2);fflush(stdout);
	//printf("\nMinimum Distance = %.3f\nPoints P[%d] : (%.3f,%.3f) and P[%d] : (%.3f,%.3f)\n",
	//		sqrt(min_dist),p1,P[p1].x,P[p1].y,p2,P[p2].x,P[p2].y);fflush(stdout);
	return sqrt(min_dist);
}

int main (void){

	time_t start, t = time(NULL); srand(t);
	float timeTaken, min_dist;
	int k, i, N=2*1048576;

	for(k=0;k<1;k++){
		//create  1D array of 2 x #points (x,y)
		point *points = (point*)malloc (sizeof(*points) * N);
		if(points==NULL){
			free(points);
			printf("Memory allocation failed while allocating for array[].\n");
			exit(1);
		}
		//i for every point(x, y)
		for(i=0;i<N;i++){
			//initialize array
			//points[i].x = ((float)rand() / ((float)RAND_MAX)*1000);
			//points[i].y = ((float)rand() / ((float)RAND_MAX)*1000);
			points[i].x = ((float) (N)/ (float)i );
			points[i].y = ((float) (N)/ (float)i );
		}

		//print_array(N,points);

		/* Start timer */
		start = clock();

		min_dist = compare_points_BF(N,points);

		/* Stop timer */
		timeTaken = (float)(clock() - start) / CLOCKS_PER_SEC;
		printf("min_dist = %f\tN = %d\tRun time: %.6f sec\n",min_dist, N, timeTaken);

		N *= 2;
		free(points);
	}
	system("PAUSE");
	return 0;
}
