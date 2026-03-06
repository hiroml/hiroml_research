#include <iostream>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <bits/stdc++.h>

// エラーチェック関数（省略せずに記述）
void checkCuda(cudaError_t err) { if (err != cudaSuccess) { std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; exit(1); } }
void checkCublas(cublasStatus_t stat) { if (stat != CUBLAS_STATUS_SUCCESS) { std::cerr << "cuBLAS Error: " << stat << std::endl; exit(1); } }

int main() {
    int size;
    std::cin >> size;
    int m = size, n = size, k = size; // サイズを大きくすると性能が出やすい
    cublasLtHandle_t ltHandle;
    checkCublas(cublasLtCreate(&ltHandle));

    // メモリ確保
    int8_t *d_a, *d_b;
    int32_t *d_c;
    checkCuda(cudaMalloc(&d_a, (size_t)m * k));
    checkCuda(cudaMalloc(&d_b, (size_t)k * n));
    checkCuda(cudaMalloc(&d_c, (size_t)m * n * 4));

    // デスクリプタ設定
    cublasLtMatmulDesc_t opDesc;
    checkCublas(cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32I, CUDA_R_32I));
    cublasLtMatrixLayout_t adesc, bdesc, cdesc;
    checkCublas(cublasLtMatrixLayoutCreate(&adesc, CUDA_R_8I, m, k, m));
    checkCublas(cublasLtMatrixLayoutCreate(&bdesc, CUDA_R_8I, k, n, k));
    checkCublas(cublasLtMatrixLayoutCreate(&cdesc, CUDA_R_32I, m, n, m));

    // --- 【重要】計測用イベントの作成 ---
    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start));
    checkCuda(cudaEventCreate(&stop));

    int32_t alpha = 1, beta = 0;

    // ウォームアップ（最初の1回は初期化コストがかかるため飛ばす）
    checkCublas(cublasLtMatmul(ltHandle, opDesc, &alpha, d_a, adesc, d_b, bdesc, &beta, d_c, cdesc, d_c, cdesc, NULL, NULL, 0, 0));
    float sumtime=0;
    // 本計測開始
    for(int i=0;i<100;i++){
    checkCuda(cudaEventRecord(start));

    checkCublas(cublasLtMatmul(ltHandle, opDesc, &alpha, d_a, adesc, d_b, bdesc, &beta, d_c, cdesc, d_c, cdesc, NULL, NULL, 0, 0));

    checkCuda(cudaEventRecord(stop));
    checkCuda(cudaEventSynchronize(stop)); // GPUの完了を待つ
    float msec = 0;
    checkCuda(cudaEventElapsedTime(&msec, start, stop));
    sumtime += msec;
    }
    
    // 性能計算 (2 * M * N * K / 時間)
    double ops = 2.0 * m * n * k;
    double tops = (ops / (sumtime / 1000.0))*100 / 1e12;

    std::cout << "Matrix Size: " << m << "x" << n << "x" << k << std::endl;
    std::cout << "Execution Time: " << sumtime << " ms" << std::endl;
    std::cout << "Performance: " << tops << " TOPS (Int8)" << std::endl;

    // 後片付け
    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
}