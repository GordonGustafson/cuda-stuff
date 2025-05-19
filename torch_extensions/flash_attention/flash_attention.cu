#include <cuda_runtime.h>
#include <stdio.h>
#include <limits>
#include <iostream>
#include <cmath>
#include <algorithm>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>


#include <Python.h>

extern "C" {
  /* Creates a dummy empty _C module that can be imported from Python.
    The import from Python will load the .so consisting of this file
    in this extension, so that the TORCH_LIBRARY static initializers
    below are run. */
  PyObject* PyInit__C(void)
  {
      static struct PyModuleDef module_def = {
          PyModuleDef_HEAD_INIT,
          "_C",   /* name of module */
          NULL,   /* module documentation, may be NULL */
          -1,     /* size of per-interpreter state of the module,
                    or -1 if the module keeps state in global variables. */
          NULL,   /* methods */
      };
      return PyModule_Create(&module_def);
  }
}


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#define CEIL_DIV(dividend, divisor) (((dividend) + (divisor) - 1) / (divisor))

__device__ static inline float onlineSoftmaxSum(float const maxA,
                                                float const sumA,
                                                float const maxB,
                                                float const sumB) {
    if (sumA == 0.0f) {
        return sumB;
    } else if (sumB == 0.0f) {
        return sumA;
    } else if (maxA > maxB) {
        return sumB * expf(maxB - maxA) + sumA;
    } else {
        return sumB + sumA * expf(maxA - maxB);
    }
}

__global__ void flash_attention_kernel(float const* const Q_HBM,  // size Mxd
                                       float const* const K_HBM,  // size Nxd
                                       float const* const V_HBM,  // size Nxd
                                       float* const O_HBM,        // size Mxd
                                       int const M,
                                       int const N,
                                       int const d,
                                       float const temperature,
                                       float* const row_sum_HBM,
                                       float* const row_max_HBM,
                                       int const maxSharedMemory) {
    extern __shared__ float sharedMemory[];
    int const B_c = min(CEIL_DIV(maxSharedMemory, 4 * d * sizeof(float)), (unsigned long)N);
    int const B_r = min(CEIL_DIV(maxSharedMemory, 4 * d * sizeof(float)), (unsigned long)d);
    int const T_c = CEIL_DIV(N, B_c);

    float* const Q = sharedMemory;
    float* const K = Q + B_r * d;
    float* const V = K + B_c * d;
    float* const S = V + B_c * d;

    // Initialize S, using threadIdx.x as the B_c dimension.
    for (int B_r_index = 0; B_r_index < B_r; B_r_index++) {
        S[B_r_index * B_c + threadIdx.x] = 0.0f;
    }

    // Load Q, using threadIdx.x to help along the d dimension
    for (int d_index = threadIdx.x; d_index < d; d_index += blockDim.x) {
        for (int B_r_index = 0; B_r_index < B_r; B_r_index++) {
            int const row_index = blockIdx.x * B_r + B_r_index;
            if (row_index < M) {
                Q[B_r_index * d + d_index] = Q_HBM[row_index * d + d_index];
            }
        }
    }

    // Iterate horizontally through different S blocks.
    for (int T_c_index = 0; T_c_index < T_c; T_c_index++) {
        // Load K and V
        for (int d_index = threadIdx.x; d_index < d; d_index += blockDim.x) {
            for (int B_c_index = 0; B_c_index < B_c; B_c_index++) {
                int const row_index = T_c_index * B_c + B_c_index;
                if (row_index < N) {
                    K[B_c_index * d + d_index] = K_HBM[row_index * d + d_index];
                    V[B_c_index * d + d_index] = V_HBM[row_index * d + d_index];
                }
            }
        }

        __syncthreads();

        // Iterate vertically within the S block.
        for (int B_r_index = 0; B_r_index < B_r; B_r_index++) {
            float S_val_for_thread = 0.0f;
            for (int d_index = 0; d_index < d; d_index++) {
                S_val_for_thread += Q[B_r_index * d + d_index] * K[threadIdx.x * d + d_index];
            }
            S[B_r_index * B_c + threadIdx.x] = S_val_for_thread / temperature;

            int const row_index = blockIdx.x * B_r + B_r_index;
            float const S_row_old_global_max = (row_index < M) ? row_max_HBM[row_index] : -INFINITY;
            float const S_row_old_global_sum = (row_index < M) ? row_sum_HBM[row_index] : 0.0f;
            __syncthreads();

            // Update max and sum for this row.
            if (threadIdx.x == 0) {
                float S_row_local_max = -INFINITY;
                float S_row_local_sum = 0.0f;
                for (int col = 0; col < B_c; col++) {
                    float const S_val_iter = S[B_r_index * B_c + col];
                    S_row_local_sum = onlineSoftmaxSum(S_row_local_max, S_row_local_sum, S_val_iter, 1.0f);
                    S_row_local_max = max(S_row_local_max, S_val_iter);
                }
                if (row_index < M) {
                    row_sum_HBM[row_index] = onlineSoftmaxSum(S_row_old_global_max,
                                                              S_row_old_global_sum,
                                                              S_row_local_max,
                                                              S_row_local_sum);
                    row_max_HBM[row_index] = max(S_row_old_global_max, S_row_local_max);
                }
            }
            __syncthreads();
            float const S_row_new_global_max = (row_index < M) ? row_max_HBM[row_index] : -INFINITY;
            float const S_row_new_global_sum = (row_index < M) ? row_sum_HBM[row_index] : 0.0f;

            // Compute P and O
            for (int d_index = threadIdx.x; d_index < d; d_index += blockDim.x) {
                float PV_val = 0.0f;
                for (int V_B_c_index = 0; V_B_c_index < B_c; V_B_c_index++) {
                    float const S_val = S[B_r_index * B_c + V_B_c_index];
                    float const P_val = expf(S_val - S_row_new_global_max) / S_row_new_global_sum;
                    PV_val += P_val * V[V_B_c_index * d + d_index];
                }

                int const row_index = blockIdx.x * B_r + B_r_index;
                if (row_index < M) {
                    int const OIndexForThread = row_index * d + d_index;
                    O_HBM[OIndexForThread] = O_HBM[OIndexForThread] * expf(S_row_old_global_max - S_row_new_global_max) * (S_row_old_global_sum / S_row_new_global_sum) + PV_val;
                }
            }
        }
    }
}


// Q, K, V, output are device pointers
void call_flash_attention_kernel(float const* const Q,  // size Mxd
                                 float const* const K,  // size Nxd
                                 float const* const V,  // size Nxd
                                 float* const output,   // size Mxd
                                 int const M,
                                 int const N,
                                 int const d) {
    int maxSharedMemory;
    gpuErrchk(cudaDeviceGetAttribute(&maxSharedMemory, cudaDevAttrMaxSharedMemoryPerBlock, 0));
    gpuErrchk(cudaFuncSetAttribute(flash_attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSharedMemory));

    int const B_c = min(CEIL_DIV(maxSharedMemory, 4 * d * sizeof(float)), (unsigned long)N);
    int const B_r = min(CEIL_DIV(maxSharedMemory, 4 * d * sizeof(float)), (unsigned long)d);
    int const T_r = CEIL_DIV(M, B_r);

    std::cout << "maxSharedMemory: " << maxSharedMemory << std::endl;
    std::cout << "B_c: " << B_c << std::endl;
    std::cout << "B_r: " << B_r << std::endl;
    std::cout << "T_r: " << T_r << std::endl;

    float* row_sum_HBM;
    gpuErrchk(cudaMalloc((void**)&row_sum_HBM, M * sizeof(float)));
    float* row_max_HBM;
    gpuErrchk(cudaMalloc((void**)&row_max_HBM, M * sizeof(float)));

    float* zeroFloats = new float[M*d]();
    gpuErrchk(cudaMemcpy(output, zeroFloats, M * d * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(row_sum_HBM, zeroFloats, M * sizeof(float), cudaMemcpyHostToDevice));

    float* negativeInfinityFloats = new float[M];
    std::fill(negativeInfinityFloats, negativeInfinityFloats + M, -INFINITY);
    gpuErrchk(cudaMemcpy(row_max_HBM, negativeInfinityFloats, M * sizeof(float), cudaMemcpyHostToDevice));

    float const temperature = sqrt(d);

    dim3 const blocksPerGrid(T_r);
    dim3 const threadsPerBlock(B_c);
    flash_attention_kernel<<<blocksPerGrid, threadsPerBlock, maxSharedMemory>>>(Q, K, V, output, M, N, d, temperature, row_sum_HBM, row_max_HBM, maxSharedMemory);
    gpuErrchk(cudaPeekAtLastError());

    float* rowSum = new float[M]();
    gpuErrchk(cudaMemcpy(rowSum, row_sum_HBM, M * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "rowSum[0]: " << row_sum_HBM[0] << std::endl;
    std::cout << "rowMax[0]: " << row_max_HBM[0] << std::endl;

    delete[] zeroFloats;
    delete[] negativeInfinityFloats;
}



torch::Tensor flash_attention_wrapper(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    TORCH_CHECK(Q.is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(K.is_cuda(), "K must be a CUDA tensor");
    TORCH_CHECK(V.is_cuda(), "V must be a CUDA tensor");

    TORCH_CHECK(Q.dtype() == torch::kFloat32, "Q must be float32");
    TORCH_CHECK(K.dtype() == torch::kFloat32, "K must be float32");
    TORCH_CHECK(V.dtype() == torch::kFloat32, "V must be float32");

    TORCH_CHECK(Q.dim() == 2, "Q must be a 2D tensor");
    TORCH_CHECK(K.dim() == 2, "K must be a 2D tensor");
    TORCH_CHECK(V.dim() == 2, "V must be a 2D tensor");

    int M = Q.size(0);
    int N = K.size(0);
    int d = Q.size(1);

    TORCH_CHECK(K.size(1) == d, "K must have the same feature dimension as Q");
    TORCH_CHECK(V.size(0) == N, "V must have the same sequence length as K");
    TORCH_CHECK(V.size(1) == d, "V must have the same feature dimension as Q");

    torch::Tensor output = torch::empty({M, d}, Q.options());

    // Call the kernel launcher
    call_flash_attention_kernel(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N, d
    );

    return output;
}

TORCH_LIBRARY(flash_attention, m) {
   // Note that "float" in the schema corresponds to the C++ double type
   // and the Python float type.
   m.def("flash_attention_wrapper(Tensor a, Tensor b, Tensor c) -> Tensor");
 }

TORCH_LIBRARY_IMPL(flash_attention, CPU, m) {
  m.impl("flash_attention_wrapper", &flash_attention_wrapper);
}

TORCH_LIBRARY_IMPL(flash_attention, CUDA, m) {
  m.impl("flash_attention_wrapper", &flash_attention_wrapper);
}
