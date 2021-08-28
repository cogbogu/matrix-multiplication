/* objective
 *      C = A*B  // A[m][k], B[k][n], C[m][n]
 * compile: nvcc -O3 matmul_double.cpp kernel_nt.cu -o matmul_double
 */

#include <iostream>
#include <cstdlib>
#include <cmath>
#include "kernel_nt.h"


void init (double *A, double *B, int M , int N, int K)
{
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < K; ++j)
        {
            A[i * K + j] = i * K + j;
            //A[i * K + j] = rand();
        }
    }

    for (int i = 0; i < K; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            B[i * N + j] = i * N + j + 1;
            //B[i * N + j] = rand();
        }
    }

}


void matmul_double_host(double* A, double* B, double* C, int M, int N, int K)
{
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            double tmp = 0;

            for (int k = 0; k < K; ++k)
            {
                tmp += A[i * K + k] * B[k * N + j];
            }

            C[i * N + j] = tmp;
        }
    }
}

void validate (double *host, double *gpu, int M, int N)
{
    bool error=false;
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            if(std::fabs(host[i * N + j] - gpu[i * N + j]) > 1e-3)
            {
                std::cerr << "possible error at position " << i << ',' << j << " host: " << host[i * N + j] << " device " << gpu[i * N + j] << '\n';
                error = true;
            }

        }
    }

    if(!error)
        std::cout<<"Validation successful\n";
}


int main(int argc, char *argv[])
{
    if(argc < 3)
    {
        std::cerr << "Usage: ./matmul_double M N K\n";
        exit(-1);
    }

    int M = std::atoi(argv[1]);
    int N = std::atoi(argv[2]);
    int K = std::atoi(argv[3]);

    /* Host alloc */
    double *hA = (double*) malloc (M * K * sizeof(double));
    double *hB = (double*) malloc (K * N * sizeof(double));
    double *hC = (double*) malloc (M * N * sizeof(double));
    double *dtohC = (double*) malloc (M * N * sizeof(double));

    /* Initialize host memory*/
    init(hA, hB, M, N, K);

    /* host compute */
    matmul_double_host(hA, hB, hC, M, N, K);

    /* device compute wrapper. Define in kernel_nt.cu */
    matmul_double_device(hA, hB, hC, M, N, K,dtohC);

    /* host vs device validation */
    validate(hC, dtohC, M, N);

    /* be clean */
    free(hA);
    free(hB);
    free(hC);
    free(dtohC);

    return 0;
}




                               

