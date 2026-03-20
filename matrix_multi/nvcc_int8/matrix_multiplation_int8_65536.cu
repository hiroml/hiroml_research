#include <iostream>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <fstream>
#include <bits/stdc++.h>
#define rep(i, n) for (int i = 3; i < (n); ++i)

using namespace std;

void write_file(vector<string> input)
{
    ofstream writing_file;
    string file_name = "./output/test.txt";
    writing_file.open(file_name);
    for (auto &i : input)
    {
        writing_file << i << endl;
    }
    writing_file.close();

    cout << "complete write" << endl;
}
int main()
{
    vector<string> S;
    int magicnum = 16;
    // サイズ
    int m = pow(2, magicnum), n = pow(2, magicnum), k = pow(2, magicnum); // サイズを大きくすると性能が出やすい

    // メモリ確保
    int8_t *d_a, *d_b;
    int32_t *d_c;
    cudaMalloc(&d_a, (sizeof(int8_t) * m * k));
    cudaMalloc(&d_b, (sizeof(int8_t)) * k * n);
    cudaMalloc(&d_c, (sizeof(int8_t) * m * n * 4));

    // 作業用メモリの確保
    size_t workspaceSize = 32 * 1024 * 1024;
    void *workspace = nullptr;
    // ハンドルの作成
    cublasLtHandle_t ltHandle;
    cublasLtCreate(&ltHandle);

    // デスクリプタ設定
    cublasLtMatmulDesc_t opDesc;
    cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32I, CUDA_R_32I);
    // レイアウトの作成
    cublasLtMatrixLayout_t adesc, bdesc, cdesc;
    cublasLtMatrixLayoutCreate(&adesc, CUDA_R_8I, m, k, m);
    cublasLtMatrixLayoutCreate(&bdesc, CUDA_R_8I, k, n, k);
    cublasLtMatrixLayoutCreate(&cdesc, CUDA_R_32I, m, n, m);

    // アルゴリズム選択
    cublasLtMatmulPreference_t preference;
    cublasLtMatmulPreferenceCreate(&preference);
    cublasLtMatmulPreferenceSetAttribute(preference,
                                         CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                         &workspaceSize, sizeof(workspaceSize));
    // 最良アルゴリズムを最大 1 件取得
    cublasLtMatmulHeuristicResult_t heuristicResult;
    int returnedResults = 0;
    cublasLtMatmulAlgoGetHeuristic(
        ltHandle, opDesc,
        adesc, bdesc, cdesc, cdesc,
        preference, 1,
        &heuristicResult, &returnedResults);
    cudaMalloc(&workspace, heuristicResult.workspaceSize);

    // 計測用イベントの作成
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int32_t alpha = 1, beta = 0;

    float sumtime = 0;
    // 本計測開始

    cudaEventRecord(start);

    cublasLtMatmul(ltHandle, opDesc, &alpha, d_a, adesc, d_b, bdesc, &beta, d_c, cdesc, d_c, cdesc, NULL, NULL, 0, 0);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop); // GPUの完了を待つ
    float msec = 0;
    cudaEventElapsedTime(&msec, start, stop);
    sumtime += msec;

    // 性能計算 (2 * M * N * K / 時間)
    double ops = 2.0 * m * n * k;
    double tops = (ops / (sumtime / 1000.0)) * 100 / 1e12;

    // cout << "Matrix Size: " << m << "x" << n << "x" << k << endl;
    // cout << "Execution Time: " << sumtime << " ms" << endl;
    // cout << "Performance: " << tops << " TOPS (Int8)" << endl;
    cout << "m" << m << ", n" << n << ", k" << k << endl;
    string text = to_string(m) + ", " + to_string(n) + ", " + to_string(k) +
                  ", " + to_string(sumtime) + ", " + to_string(tops);
    S.push_back(text);
    // 後片付け
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasLtMatrixLayoutDestroy(adesc);
    cublasLtMatrixLayoutDestroy(bdesc);
    cublasLtMatrixLayoutDestroy(cdesc);
    cublasLtMatmulDescDestroy(opDesc);
    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtDestroy(ltHandle);
    cudaFree(workspace);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    write_file(S);
    return 0;
}