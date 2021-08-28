/* objective

 *      c = A*b  // A[m][n] is a matrix, b[n] and c[m] are vectors

 * compile: nvcc -O3 matvec.cu -o matvec

 */



 #include <iostream>

 #include <cstdlib>

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



 void init (double *matA3d, double *matB3d, int p, int q, int r)

 {

     for (int i = 0; i < p; ++i)

     {

         for (int j = 0; j < q; ++j)

         {
            for (int k = 0; k < r; ++k)

            {
   
                matA3d[(i * q + j) + (r * k)] = (i * q + j) * r + k;
                matB3d[(i * q + j) + (r * k)] = (i * q + j) * r + k;
   
            }

         }

     }




}





void matAdd3d_h(double* matA3d, double* matB3d, double* matC3d, int p, int q, int r)

{



        for (int i = 0; i < (p * q * r); ++i)

        {

            //vec_out[i] += mat[i * n + j] * vec_in[j];
            matC3d[i] = matA3d[i] + matB3d[i];

        }

  

}



__global__ void matAdd3d_d(double* matA3d, double* matB3d, double* matC3d, int p, int q, int r)

{

    /// complete code

   //int blckx = blockIdx.x;
   
   int bx = blockIdx.x;
   int tx = threadIdx.x;
   //__shared__ result[BLK_SIZE];

   //int trdy = threadIdx.y;
   
   int idx = (bx * BLK_SIZE) + tx;

   if(idx < (p * q * r)){
       matC3d[idx * q +)] = matB3d[idx] + matA3d[idx];
   }




   double rSum = 0;

   for(int i = 0; i < n; i++)

       rSum = rSum + (mat[r * n + i] * vec_in[i]);



   //int vecpos = r * n;

   vec_out[r] = rSum;





}



void validate (double *host, double *gpu, int m)

{



    for (int i = 0; i < m; ++i)

    {

        if(std::abs(host[i] - gpu[i]) > 1e-3)

        {

            std::cerr << "possible error at position " << i << " host: " << host[i] << " device " << gpu[i] << '\n';

        }

    }

}


int main(int argc, char *argv[])

 {

     if(argc < 2)

     {

         std::cerr << "Usage: ./matvec M N\n";

         exit(-1);

     }



     int m = std::atoi(argv[1]);

     int n = std::atoi(argv[2]);



     /* Host alloc */

     double *mat = (double*) malloc (m * n * sizeof(double));

     double *vec_in = (double*) malloc (n * sizeof(double));

     double *vec_out = (double*) malloc (m * sizeof(double));



     /*device array pointers*/

     double *d_mat;

     double *d_vec_in;

     double *d_vec_out;



     /* Device alloc */

     /// complete code

     cudaMalloc(&d_mat, sizeof(double)*(m*n));

     cudaMalloc(&d_vec_in, sizeof(double)*(n));

     cudaMalloc(&d_vec_out, sizeof(double)*(m));


     /* Initialize host memory*/

     init(mat, vec_in, m, n);



     /* host compute */

     matvec_h(mat, vec_in, vec_out, m, n);





     /* Copy from host to device */

     /// complete code

     cudaMemcpy(d_mat, mat, sizeof(double)*(m * n ), cudaMemcpyHostToDevice);

     cudaMemcpy(d_vec_in, vec_in, sizeof(double)*(n), cudaMemcpyHostToDevice);

     cudaMemcpy(d_vec_out, vec_out, sizeof(double)*(n), cudaMemcpyHostToDevice);



    /* create block and grid size*/

    dim3 threads(128);

    dim3 grid((int)ceil((float)m/threads.x));





     /* call gpu kernel */

     /// complete code

     matvec_d<<<grid, threads>>>(d_mat, d_vec_in, d_vec_out, m, n);





     /* Copy from device to host */

     /// complete code
     double *gpu_result = (double*) malloc (m * sizeof(double));

     cudaMemcpy(gpu_result, d_vec_out, sizeof(double)*(m), cudaMemcpyDeviceToHost);



     /* host vs device validation */

     /// REPLACE one vec_out with the result array that you moved from device to host

     validate(gpu_result, vec_out, m);





     /* be clean */

     free(mat);

     free(vec_in);

     free(vec_out);



     /// add code to free gpu memory

     cudaFree(d_mat);

     cudaFree(d_vec_in);

     cudaFree(d_vec_out);



     return 0;

 }



