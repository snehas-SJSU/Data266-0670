/*
 * DATA 266 HW1 — C = A * B (square matrices)
 * SID4 = 0670
 *
 * How blocks and threads work here:
 *   I launch a 2D grid of blocks. Each block is 16 x 16 threads.
 *   thread (threadIdx.x, threadIdx.y) inside block (blockIdx.x, blockIdx.y)
 *   computes ONE output:
 *      row = blockIdx.y * 16 + threadIdx.y
 *      col = blockIdx.x * 16 + threadIdx.x
 *      C[row, col] = sum_k A[row, k] * B[k, col]
 *   Extra threads (when N is not a multiple of 16) just return.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define THREADS 16

void cpu_matmul(const float *A, const float *B, float *C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float s = 0.0f;
            for (int k = 0; k < N; k++) {
                s += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = s;
        }
    }
}

__global__ void gpu_matmul(const float *A, const float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float s = 0.0f;
        for (int k = 0; k < N; k++) {
            s += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = s;
    }
}

double ms_now(struct timespec t0, struct timespec t1) {
    return (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
}

int main(void) {
    int sizes[3] = {256, 1024, 4096};
    printf("SID4=0670  C = A * B\n");
    printf("each block is %d x %d threads\n", THREADS, THREADS);
    printf("each thread writes one C[row,col]\n\n");
    printf("%8s %12s %14s %14s %10s\n", "N", "CPU_ms", "kernel_ms", "H2D+D2H_ms", "speedup");

    for (int t = 0; t < 3; t++) {
        int N = sizes[t];
        size_t bytes = (size_t)N * (size_t)N * sizeof(float);

        float *A = (float *)malloc(bytes);
        float *B = (float *)malloc(bytes);
        float *Ccpu = (float *)malloc(bytes);
        float *Cgpu = (float *)malloc(bytes);
        for (int i = 0; i < N * N; i++) {
            A[i] = (i % 13) * 0.01f;
            B[i] = (i % 7) * 0.02f;
        }

        // CPU baseline
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        cpu_matmul(A, B, Ccpu, N);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double cpu_ms = ms_now(t0, t1);

        float *dA, *dB, *dC;
        cudaMalloc((void **)&dA, bytes);
        cudaMalloc((void **)&dB, bytes);
        cudaMalloc((void **)&dC, bytes);

        dim3 threads(THREADS, THREADS);
        dim3 blocks((N + THREADS - 1) / THREADS, (N + THREADS - 1) / THREADS);

        // warmup so the timed run is not the first launch
        cudaMemcpy(dA, A, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(dB, B, bytes, cudaMemcpyHostToDevice);
        gpu_matmul<<<blocks, threads>>>(dA, dB, dC, N);
        cudaDeviceSynchronize();

        cudaEvent_t e0, e1, e2, e3;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        cudaEventCreate(&e2);
        cudaEventCreate(&e3);

        cudaEventRecord(e0);
        cudaMemcpy(dA, A, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(dB, B, bytes, cudaMemcpyHostToDevice);
        cudaEventRecord(e1);
        gpu_matmul<<<blocks, threads>>>(dA, dB, dC, N);
        cudaEventRecord(e2);
        cudaMemcpy(Cgpu, dC, bytes, cudaMemcpyDeviceToHost);
        cudaEventRecord(e3);
        cudaEventSynchronize(e3);

        float h2d = 0, kern = 0, d2h = 0;
        cudaEventElapsedTime(&h2d, e0, e1);
        cudaEventElapsedTime(&kern, e1, e2);
        cudaEventElapsedTime(&d2h, e2, e3);
        float xfer = h2d + d2h;
        float e2e = h2d + kern + d2h;
        float speedup = (float)(cpu_ms / e2e);

        printf("%8d %12.3f %14.3f %14.3f %10.3f\n", N, cpu_ms, kern, xfer, speedup);

        if (N == 256) {
            float maxdiff = 0.0f;
            for (int i = 0; i < N * N; i++) {
                float d = fabsf(Ccpu[i] - Cgpu[i]);
                if (d > maxdiff) maxdiff = d;
            }
            printf("  N=256 max |CPU-GPU| = %g\n", maxdiff);
        }

        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
        free(A);
        free(B);
        free(Ccpu);
        free(Cgpu);
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        cudaEventDestroy(e2);
        cudaEventDestroy(e3);
    }
    return 0;
}
