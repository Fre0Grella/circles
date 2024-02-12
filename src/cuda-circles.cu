/***
 % Galeri Marco <marco.galeri@studio.unibo.it> 
 % 0001019991
***/

/****************************************************************************
 *
 * circles.c - Circles intersection
 *
 * Copyright (C) 2023 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 ****************************************************************************/

/***
% Circles intersection
% Moreno Marzolla <moreno.marzolla@unibo.it>

This is a parallelized implementation of the circle intersection program
described in the specification using CUDA.

To compile:

        nvcc cuda-circles.cu -o cuda-circles.cu -lm

To execute:

        ./cuda-circles [ncircles [iterations]]

where `ncircles` is the number of circles, and `iterations` is the
number of iterations to execute.
***/

#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

/* define the dimension of thread-block considering a bidimensional structure */
#define BLKDIM 32

typedef struct {
    float x, y;   /* coordinates of center */
    float r;      /* radius */
    float dx, dy; /* displacements due to interactions with other circles */
} circle_t;

/* These constants can be replaced with #define's if necessary */
const float XMIN = 0.0;
const float XMAX = 1000.0;
const float YMIN = 0.0;
const float YMAX = 1000.0;
const float RMIN = 10.0;
const float RMAX = 100.0;
const float EPSILON = 1e-5;
const float K = 1.5;
/* Initialize these constant in the device memory */
__constant__ float D_EPSILON;
__constant__ float D_K;

int ncircles;
circle_t *circles = NULL;

/**
 * Return a random float in [a, b]
 */
float randab(float a, float b)
{
    return a + (((float)rand())/RAND_MAX) * (b-a);
}

/**
 * Create and populate the array `circles[]` with randomly placed
 * circls.
 *
 * Do NOT parallelize this function.
 */
void init_circles(int n)
{
    assert(circles == NULL);
    ncircles = n;
    circles = (circle_t*)malloc(n * sizeof(*circles));
    assert(circles != NULL);
    for (int i=0; i<n; i++) {
        circles[i].x = randab(XMIN, XMAX);
        circles[i].y = randab(YMIN, YMAX);
        circles[i].r = randab(RMIN, RMAX);
        circles[i].dx = circles[i].dy = 0.0;
    }
}


/**
 * Set all displacements to zero.
 */
__global__
void reset_displacements_kernel( circle_t* d_circles, int ncircles )
{   
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < ncircles) {
        d_circles[i].dx = d_circles[i].dy = 0.0;
    }
}

/**
 * Compute the force acting on each circle; storing the number of
 * overlapping pairs of circles in the device memory (each overlapping pair must be counted
 * only once).
 */
__global__
void compute_forces_kernel( circle_t* circles, int ncircles, int* n_intersection)
{
    /* Assigning an index of the circles array for every thread used. */
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    /*  This condition only allowed the necessary thread 
        and avoid access to out of bound memory. */
    if (i < ncircles && i < j && j < ncircles) {
        
        const float deltax = circles[j].x - circles[i].x;
        const float deltay = circles[j].y - circles[i].y;
        /* hypotf(x,y) computes sqrtf(x*x + y*y) avoiding
           overflow. This function is defined in <math.h>, and
           should be available also on CUDA. In case of troubles,
           it is ok to use sqrtf(x*x + y*y) instead. */
        const float dist = hypotf(deltax, deltay);
        const float Rsum = circles[i].r + circles[j].r;
        
        if (dist < Rsum - D_EPSILON) {
            /* avoid race condition using atomic operation. */
            atomicAdd(n_intersection,1);
            
            const float overlap = Rsum - dist;
            assert(overlap > 0.0);
            // avoid division by zero
            const float overlap_x = overlap / (dist + D_EPSILON) * deltax;
            const float overlap_y = overlap / (dist + D_EPSILON) * deltay;
            
            /* avoid race condition using atomic operation */
            atomicAdd(&circles[i].dx, -overlap_x / D_K);
            atomicAdd(&circles[i].dy, -overlap_y / D_K);
            atomicAdd(&circles[j].dx, overlap_x / D_K);
            atomicAdd(&circles[j].dy, overlap_y / D_K);
        }
    }
}

/**
 * Move the circles to a new position according to the forces acting
 * on each one.
 */
__global__
void move_circles_kernel( circle_t* circles, int ncircles )
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < ncircles) {
        circles[i].x += circles[i].dx;
        circles[i].y += circles[i].dy;
    }  
}

#ifdef MOVIE
/**
 * Dumps the circles into a text file that can be processed using
 * gnuplot. This function may be used for debugging purposes, or to
 * produce a movie of how the algorithm works.
 *
 * You may want to completely remove this function from the final
 * version.
 */
void dump_circles( int iterno )
{
    char fname[64];
    snprintf(fname, sizeof(fname), "circles-%05d.gp", iterno);
    FILE *out = fopen(fname, "w");
    const float WIDTH = XMAX - XMIN;
    const float HEIGHT = YMAX - YMIN;
    fprintf(out, "set term png notransparent large\n");
    fprintf(out, "set output \"circles-%05d.png\"\n", iterno);
    fprintf(out, "set xrange [%f:%f]\n", XMIN - WIDTH*.2, XMAX + WIDTH*.2 );
    fprintf(out, "set yrange [%f:%f]\n", YMIN - HEIGHT*.2, YMAX + HEIGHT*.2 );
    fprintf(out, "set size square\n");
    fprintf(out, "plot '-' with circles notitle\n");
    for (int i=0; i<ncircles; i++) {
        fprintf(out, "%f %f %f\n", circles[i].x, circles[i].y, circles[i].r);
    }
    fprintf(out, "e\n");
    fclose(out);
}
#endif

int main( int argc, char* argv[] )
{
    /* Initialize device constant. */
    cudaMemcpyToSymbol(D_EPSILON, &EPSILON, sizeof(float));
    cudaMemcpyToSymbol(D_K, &K, sizeof(float));
    int n = 10000;
    int iterations = 20;

    if ( argc > 3 ) {
        fprintf(stderr, "Usage: %s [ncircles [iterations]]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if (argc > 1) {
        n = atoi(argv[1]);
    }

    if (argc > 2) {
        iterations = atoi(argv[2]);
    }

    init_circles(n);

    circle_t* d_circles;    /* device pointer of circles */
    size_t size_circles = ncircles * sizeof(circle_t);  /* size of circles*/
    int* d_overlaps;    /* device pointer of the no. of overlaps */
    int h_overlaps; /* host memory of the no. of overlaps*/
    int grid1d = (ncircles + 1023) / BLKDIM * BLKDIM;   /* size of the grid in 1 dimension*/
    dim3 grid2d((ncircles + BLKDIM-1) / BLKDIM, (ncircles + BLKDIM-1) / BLKDIM);    /* size of the grid in 2 dimension */
    dim3 block(BLKDIM,BLKDIM);  /* size of blocks in 2 dimension */

    /* Allocate memory for the device pointer */
    cudaSafeCall(cudaMalloc((void**)&d_circles, size_circles));
    cudaSafeCall(cudaMemcpy(d_circles, circles, size_circles, cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMalloc((void**)&d_overlaps, sizeof(int)));

    const double tstart_prog = hpc_gettime();
#ifdef MOVIE
    dump_circles(0);
#endif
/*  Even if achieve minimal to no benefit this directive
    unroll the following loop reducing loop overhead. */
#pragma unroll
     for (int it=0; it<iterations; it++) 
    {
        /*  After every iteration reset the number of overlaps 
            and copy it in to the device memory. */
        h_overlaps = 0; 
        cudaSafeCall(cudaMemcpy(d_overlaps, &h_overlaps, sizeof(int), cudaMemcpyHostToDevice));

        const double tstart_iter = hpc_gettime();
        /* Call this kernel function with a 1 dimensional structure. */
        reset_displacements_kernel<<<grid1d,BLKDIM*BLKDIM>>>(d_circles, ncircles);
        cudaCheckError();
        /* Call this kernel function with a 2 dimensional structure. */
        compute_forces_kernel<<<grid2d,block>>>(d_circles, ncircles, d_overlaps);
        cudaCheckError();
        /* Return the number of overlaps in the host memory. */
        cudaSafeCall(cudaMemcpy(&h_overlaps,d_overlaps,sizeof(int),cudaMemcpyDeviceToHost));
        /* Call this kernel function with a 1 dimensional structure. */
        move_circles_kernel<<<grid1d,BLKDIM*BLKDIM>>>(d_circles, ncircles);
        cudaCheckError();
        const double elapsed_iter = hpc_gettime() - tstart_iter;
#ifdef MOVIE
        dump_circles(it+1);
#endif
        printf("Iteration %d of %d, %d overlaps (%f s)\n", it + 1, iterations, h_overlaps, elapsed_iter);  
    }
    const double elapsed_prog = hpc_gettime() - tstart_prog;
    printf("Elapsed time: %f\n", elapsed_prog);
    float throughput = ( (n*(n+1)/2)*iterations ) / (1e9 * elapsed_prog);
    printf("Throughput: %f Gops/s\n", throughput);
    /* Free any dynamically allocated memory */
    free(circles);
    cudaFree(d_circles);
    cudaFree(d_overlaps);
    
    return EXIT_SUCCESS;
}
