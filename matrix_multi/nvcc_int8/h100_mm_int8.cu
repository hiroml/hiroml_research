#include <iostream>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <cuda_bf16.h>

using namespace std;

// エラーチェックマクロ（コードをスッキリさせるため）
#define CHECK_CUDA(err) { if (err != cudaSuccess) { cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << endl; exit(1); } }
#define CHECK_CUBLAS(stat) { if (stat != CUBLAS_STATUS_SUCCESS) { cerr << "cuBLAS Error: " << stat << " at line " << __LINE__ << endl; exit(1); } }

int main() {
    vector<string> S;
    cublasLtHandle_t ltHandle;
    CHECK_CUBLAS(cublasLtCreate(&ltHandle));

    // --- [最適化1] ワークスペースの固定確保 (H100には必須級) ---
    size_t workspaceSize = 32 * 1024 * 1024; // 32MB
    void* d_workspace;
    CHECK_CUDA(cudaMalloc(&d_workspace, workspaceSize));

    // 最大サイズを想定してメモリを事前に大きく確保（ループ内のMallocを排除）
    // ループごとにMalloc/FreeするとH100ではオーバーヘッドが無視できません
    int max_dim = pow(2, 16); 
    int8_t *d_a, *d_b;
    int32_t *d_c;
    CHECK_CUDA(cudaMalloc(&d_a, (size_t)max_dim * max_dim * sizeof(int8_t)));
    CHECK_CUDA(cudaMalloc(&d_b, (size_t)max_dim * max_dim * sizeof(int8_t)));
    CHECK_CUDA(cudaMalloc(&d_c, (size_t)max_dim * max_dim * sizeof(int32_t)));

    cublasLtMatmulDesc_t opDesc;
    // H100のBF16計算は CUBLAS_COMPUTE_32F で内部演算をTF32/FP32で行うのが標準
    CHECK_CUBLAS(cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    for (int i = 3; i < 17; ++i) {
        for (int j = 3; j < 17; ++j) {
            for (int l = 3; l < 17; ++l) {
                int m = pow(2, i), n = pow(2, j), k = pow(2, l);
                
                // --- [最適化2] レイアウトの動的更新 ---
                cublasLtMatrixLayout_t adesc, bdesc, cdesc;
                CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&adesc, CUDA_R_16BF, m, k, m));
                CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&bdesc, CUDA_R_16BF, k, n, k));
                CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&cdesc, CUDA_R_16BF, m, n, m));

                cudaEvent_t start, stop;
                CHECK_CUDA(cudaEventCreate(&start));
                CHECK_CUDA(cudaEventCreate(&stop));

                float alpha = 1.0f, beta = 0.0f;

                // ウォームアップ & ワークスペース注入
                CHECK_CUBLAS(cublasLtMatmul(ltHandle, opDesc, &alpha, d_a, adesc, d_b, bdesc, &beta, d_c, cdesc, d_c, cdesc, 
                                          NULL, d_workspace, workspaceSize, 0));

                float sumtime = 0;
                for (int iter = 0; iter < 100; iter++) {
                    CHECK_CUDA(cudaEventRecord(start));
                    CHECK_CUBLAS(cublasLtMatmul(ltHandle, opDesc, &alpha, d_a, adesc, d_b, bdesc, &beta, d_c, cdesc, d_c, cdesc, 
                                              NULL, d_workspace, workspaceSize, 0));
                    CHECK_CUDA(cudaEventRecord(stop));
                    CHECK_CUDA(cudaEventSynchronize(stop));
                    float msec = 0;
                    CHECK_CUDA(cudaEventElapsedTime(&msec, start, stop));
                    sumtime += msec;
                }

                double ops = 2.0 * m * n * k;
                double tops = (ops / (sumtime / 1000.0)) * 100 / 1e12;

                cout << "m" << m << ", n" << n << ", k" << k << " -> " << tops << " TFLOPS" << endl;
                S.push_back(to_string(m) + "," + to_string(n) + "," + to_string(k) + "," + to_string(sumtime) + "," + to_string(tops));

                CHECK_CUDA(cudaEventDestroy(start));
                CHECK_CUDA(cudaEventDestroy(stop));
                CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(adesc));
                CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(bdesc));
                CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(cdesc));
            }
        }
    }

    // 後片付け
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); cudaFree(d_workspace);
    cublasLtMatmulDescDestroy(opDesc);
    cublasLtDestroy(ltHandle);
    
    return 0;
}