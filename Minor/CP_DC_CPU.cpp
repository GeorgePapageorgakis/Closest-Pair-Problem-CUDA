#include <stdio.h>
#include <fstream>
#include <stdlib.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <string.h>
#include <time.h>

using namespace std;

typedef struct { float x, y; } point_t, *point;

unsigned time_seed(){
	time_t now = time ( 0 );
	unsigned char *p = (unsigned char *)&now;
	unsigned seed = 0;
	size_t i;

	for ( i = 0; i < sizeof now; i++ )
		seed = seed * ( UCHAR_MAX + 2U ) + p[i];

	return seed;
}
 
inline float dist(point a, point b) {
	float dx = a->x - b->x, dy = a->y - b->y;
	return dx * dx + dy * dy;
}
 
inline int cmp_dbl(float a, float b) {
    return a < b ? -1 : a > b ? 1 : 0;
}
 
int cmp_x(const void *a, const void *b) {
	return cmp_dbl( (*(const point*)a)->x, (*(const point*)b)->x );
}
 
int cmp_y(const void *a, const void *b) {
	return cmp_dbl( (*(const point*)a)->y, (*(const point*)b)->y );
}
 
float brute_force(point* pts, int max_n, point *a, point *b) {
	int i, j;
	float d, min_d = FLT_MAX;

	for (i = 0; i < max_n; i++) {
		//printf("point[%d].x = %f, point[%d].y = %f\n",  i, pts[i]->x, i, pts[i]->y);
		for (j = i + 1; j < max_n; j++) {
			d = dist(pts[i], pts[j]);
			if (d >= min_d ) continue;
			*a = pts[i];
			*b = pts[j];
			min_d = d;
		}
	}
	return min_d;
}
 
float closest(point* sx, int nx, point* sy, int ny, point *a, point *b) {
	int left, right, i;
	float d, min_d, x0, x1, mid, x;
	point a1, b1;
	point *s_yy;

	if (nx <= 8) return brute_force(sx, nx, a, b);

	s_yy = (point*) malloc(sizeof(point) * ny);
	mid = sx[nx/2]->x;

	/* adding points to the y-sorted list; if a point's x is less than mid,
	   add to the begining; if more, add to the end backwards, hence the
	   need to reverse it (more like sorting each batch S_x as of y)*/
	left = -1; right = ny;
	for (i = 0; i < ny; i++)
		if (sy[i]->x < mid) s_yy[ ++ left ] = sy[i];
		else                s_yy[ -- right] = sy[i];

	/* reverse the higher part of the list */
	for (i = ny - 1; right < i; right ++, i--) {
		a1 = s_yy[right]; s_yy[right] = s_yy[i]; s_yy[i] = a1;
	}

	min_d = closest(sx, nx/2, s_yy, left + 1, a, b);
	d = closest(sx + nx/2, nx - nx/2, s_yy + left + 1, ny - left - 1, &a1, &b1);

	if (d < min_d) { min_d = d; *a = a1; *b = b1; }
	d = sqrt(min_d);

	/* get all the points within distance d of the center line */
	left = -1; right = ny;
	for (i = 0; i < ny; i++) {
		x = sy[i]->x - mid; //sy cause its already sorted as of y
		if (x <= -d || x >= d) continue;

		if (x < 0) s_yy[++left]  = sy[i];
		else       s_yy[--right] = sy[i];
	}

	/* compare each left point to right point */
	while (left >= 0) {
		x0 = s_yy[left]->y + d;

		while (right < ny && s_yy[right]->y > x0) right ++;
		if (right >= ny) break;

		x1 = s_yy[left]->y - d;
		for (i = right; i < ny && s_yy[i]->y > x1; i++)
			if ((x = dist(s_yy[left], s_yy[i])) < min_d) {
				min_d = x;
				d = sqrt(min_d);
				*a = s_yy[left];
				*b = s_yy[i];
			}

		left --;
	}

	free(s_yy);
	return min_d;
}
 
//Number of total points power of 2
#define NP 1048576

int main(void)
{
	time_t start, stop;
	float time, distance;
	int i;
	point a, b;

	srand ( time_seed() );

	point pts  = (point) malloc(sizeof(point_t) * NP);
	point* s_x = (point*) malloc(sizeof(point) * NP);
	point* s_y = (point*) malloc(sizeof(point) * NP);

	//initialize P array in host from file
    ifstream file_reader_x( "E:\\rand_x_num.txt" );
	if ( ! file_reader_x.is_open())
		printf("Could not open rand_x_num file!\n");
	
	ifstream file_reader_y( "E:\\rand_y_num.txt" );
	if ( ! file_reader_y.is_open())
		printf("Could not open rand_y_num file!\n");

	for ( int i = 0; i < NP; i++ ){
		float pt;
		s_x[i] = pts + i;
		file_reader_x >> pt;		pts[i].x = pt;
		file_reader_y >> pt;		pts[i].y = pt;
		//pts[i].x = ((float)rand() / ((float)RAND_MAX));
		//pts[i].y = ((float)rand() / ((float)RAND_MAX));
		//pts[i].x = ((float) (NP)/ (float)i );
		//pts[i].y = ((float) (NP)/ (float)i );
		//pts[i].x = ((float) i/ (float)(NP) );
		//pts[i].y = ((float) i/ (float)(NP) );
	}

	/*
	start = clock();
    printf("brute force: %g, ", sqrt(brute_force(s_x, NP, &a, &b)));
	printf("between (%f,%f) and (%f,%f)\n", a->x, a->y, b->x, b->y);
	printf("Run time: %.6f sec\n", (float)(clock() - start) / CLOCKS_PER_SEC);
	*/
	//start = clock();
	memcpy(s_y, s_x, sizeof(point) * NP);
	start = clock();
	qsort(s_x, NP, sizeof(point), cmp_x);
	qsort(s_y, NP, sizeof(point), cmp_y);

	//start = clock();
	/*distance = sqrt(closest(s_x, NP, s_y, NP, &a, &b));
	stop = clock();
	time = (float)(stop - start) / CLOCKS_PER_SEC;
	printf("Run time: %.6f sec\n", time);
	*/
	//start = clock();
	printf("min = %.24f \n", sqrt(closest(s_x, NP, s_y, NP, &a, &b)));
	printf("point (%.20f, %.20f)\n and (%.20f, %.20f)\n", a->x, a->y, b->x, b->y);
	printf("Run time: %.6f sec\n", (float)(clock() - start) / CLOCKS_PER_SEC);
	
	/* free memory */
	free(pts);
	free(s_x);
	free(s_y);

	getchar();
	return 0;
}