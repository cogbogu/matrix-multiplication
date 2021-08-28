#include <iostream>
#include <cstdlib>
#include "kernel_nt.h"
#define BLK_SIZE 32

#define EC(ans) { chkerr((ans), __FILE__, __LINE__); }
inline void chkerr(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess)
    {
        std::cerr << "ERROR!!!:" << cudaGetErrorString(code) << " File: " << file << " Line: " << line << '\n';
        exit(-1);
    }
}

__global__ void matmul_double(double* A, double* B , double* C, int M, int N, int K)
{
        int bx = blockIdx.x;
        int by = blockIdx.y;
        int tx = threadIdx.x;
        int ty = threadIdx.y;

        int row = by * BLK_SIZE + ty;
        int col = bx * BLK_SIZE + tx;

        __shared__ double sa[BLK_SIZE][BLK_SIZE];
        __shared__ double sb[BLK_SIZE][BLK_SIZE];

        double sum = 0;


        for(int tile = 0; tile < (BLK_SIZE + K - 1)/BLK_SIZE; tile+=1){
                if(tile * BLK_SIZE + tx < K && row < M)
                        sa[ty][tx] = A[row * K + tx + tile*BLK_SIZE];
                else
                        sa[ty][tx] = 0.0;

                if(tile * BLK_SIZE + ty < K && col < N)
                        sb[ty][tx] = B[(tile*BLK_SIZE + ty) * N + col];
                else
                        sb[ty][tx] = 0.0;

                __syncthreads();

                for(int i = 0; i < BLK_SIZE; ++i){
                        sum += sa[ty][i] * sb[i][tx];
                }
                
                __syncthreads();

        }

        if(row < M && col < N)
                C[row*N +col] = sum;

}



__global__ void matmul_double_t(double* A, double* B , double* C, int M, int N, int K){

        int bx = blockIdx.x;
        int by = blockIdx.y;
        int tx = threadIdx.x;
        int ty = threadIdx.y;

        int row = by * BLK_SIZE + ty;
        int col = bx * BLK_SIZE + tx;

        //printf("%lf,  ", B[col * N + row]);
        __shared__ double sa[BLK_SIZE][BLK_SIZE];
        __shared__ double sb[BLK_SIZE][BLK_SIZE];
        //with shared memory

        //sb[ty][tx] = B[row * N + col];
        //B[row * N + col] = sb[ty][tx];
        __syncthreads();

        double sum = 0;

        for(int tile = 0; tile < (BLK_SIZE + K - 1)/BLK_SIZE; tile+=1){
                if(tile * BLK_SIZE + tx < K && row < M)
                        sa[ty][tx] = A[row * K + tx + tile*BLK_SIZE];
                else
                        sa[ty][tx] = 0;

                if(tile * BLK_SIZE + ty < K && col < N){
                       //B[(tile*BLK_SIZE + ty) * N + col] = sb[ty][tx];
                       row = bx * BLK_SIZE + ty;
                       col = by * BLK_SIZE + tx;
                       sb[tx][ty] = B[row * N + col];
                       //sb[ty][tx] = B[row * K + tx + tile*BLK_SIZE];
                }else{
                        sb[ty][tx] = 0;
                }
                __syncthreads();

                       row = by * BLK_SIZE + ty;
                       col = bx * BLK_SIZE + tx;

                for(int i = 0; i < BLK_SIZE; i++){
                        sum += sa[ty][i] * sb[i][tx];
                }

                __syncthreads();

        }
        if(row < M && col < N)
                C[row*N + col] = sum;


}

void matmul_double_device(double* A, double* B, double* C, int M, int N, int K, double *dtohC)
{
    /// allocate GPU memory here
    double *d_A;

    double *d_B;

    double *d_C;

    size_t bytes = sizeof(double);

    cudaMalloc((void**) &d_A, bytes*(M * K));
    cudaMalloc((void**) &d_B, bytes*(K * N));
    cudaMalloc((void**) &d_C, bytes*(M * N));


    /// all host to device memcpy here
    cudaMemcpy(d_A, A, bytes*(M * K), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, bytes*(K * N), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, bytes*(M * N), cudaMemcpyHostToDevice);


    //----------- Begin kernel call for C=A*B ------------
    float time_ms;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    /// call kernel here

    dim3 threads(BLK_SIZE, BLK_SIZE);
    dim3 grid((int) ceil((float) M/BLK_SIZE), (int) ceil((float)K/BLK_SIZE));

    matmul_double<<<grid, threads>>>(d_A, d_B, d_C, M, N, K);
    //matmul_double_t<<<grid, threads>>>(d_A, d_B, d_C, M, N, K);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);

    //compute GFLOPS
        double gflop = abs(((2 * M*K*N) - (M*N)) * 1e-9);
        double op_time_s = time_ms * 1e-3;
        double gflops = gflop/op_time_s;

    printf("Kernel time :  %f ms \n", time_ms);
    printf("GFLOPS :  %f \n", gflops);

    /// all device to host memcpy here
    cudaMemcpy(dtohC, d_C, bytes * (M * N), cudaMemcpyDeviceToHost);

    /// code to free gpu memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);


}


void matmul_double_device_t(double* A, double* B, double* C, int M, int N, int K, double *dtohC)
{
    /// allocate GPU memory here
    double *d_A;

    double *d_B;

    double *d_C;

    size_t bytes = sizeof(double);

    cudaMalloc((void**) &d_A, bytes*(M * K));
    cudaMalloc((void**) &d_B, bytes*(K * N));
    cudaMalloc((void**) &d_C, bytes*(M * N));
    
    /// all host to device memcpy here
    cudaMemcpy(d_A, A, bytes*(M * K), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, bytes*(K * N), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, bytes*(M * N), cudaMemcpyHostToDevice);


    //----------- Begin kernel call for C=A*B ------------
    float time_ms;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    /// call kernel here

    dim3 threads(BLK_SIZE, BLK_SIZE);
    dim3 grid((int) ceil((float) M/BLK_SIZE), (int) ceil((float)K/BLK_SIZE));

    matmul_double_t<<<grid, threads>>>(d_A, d_B, d_C, M, N, K);
    //matmul_double_t<<<grid, threads>>>(d_A, d_B, d_C, M, N, K);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);

    //compute GFLOPS
    double gflop = abs(((2 * M*K*N) - (M*N)) * 1e-9);
    double op_time_s = time_ms * 1e-3;
    double gflops = gflop/op_time_s;

    printf("Kernel time :  %f ms \n", time_ms);
    printf("GFLOPS :  %f \n", gflops);

    /// all device to host memcpy here
    cudaMemcpy(dtohC, d_C, bytes * (M * N), cudaMemcpyDeviceToHost);

    /// code to free gpu memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);


}


