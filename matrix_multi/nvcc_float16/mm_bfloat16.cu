#include <iostream>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <fstream>
#include <bits/stdc++.h>
#include <cuda_bf16.h>


using namespace std;

void write_file(vector<string> input)
{
    ofstream writing_file;
    string file_name = "./output/h100.txt";
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
    long long m = pow(2, magicnum), n = pow(2, magicnum), k = pow(2, magicnum); // サイズを大きくすると性能が出やすい

    // メモリ確保
    __nv_bfloat16 *d_a, *d_b, *d_c;
    cudaError_t err;
    err = cudaMalloc(&d_a, sizeof(__nv_bfloat16) * m * k);
    cout << "d_a: " << cudaGetErrorString(err) << endl;
    err = cudaMalloc(&d_b, sizeof(__nv_bfloat16) * k * n);
    cout << "d_b: " << cudaGetErrorString(err) << endl;
    err = cudaMalloc(&d_c, sizeof(__nv_bfloat16) * m * n);
    cout << "d_c: " << cudaGetErrorString(err) << endl;
    // 作業用メモリの確保
    size_t workspaceSize = 32 * 1024 * 1024;
    void *workspace = nullptr;
    cudaMalloc(&workspace, workspaceSize);

    // ハンドルの作成
    cublasLtHandle_t ltHandle;
    cublasLtCreate(&ltHandle);

    // デスクリプタ設定
    //内部累積32fのため32
    cublasLtMatmulDesc_t opDesc;
    cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    
    // 計測用イベントの作成
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float alpha = 1.0, beta = 0.0;

    for (int ii = 3; ii <= magicnum; ii++)
    {
        for (int jj = 3; jj <= magicnum; jj++)
        {
            for (int ll = 3; ll <= magicnum; ll++)
            {
                float sumtime = 0;
                long long mm = pow(2, ii), nn = pow(2, jj), kk = pow(2, ll);
                // レイアウトの作成
                cublasLtMatrixLayout_t adesc, bdesc, cdesc;
                cublasLtMatrixLayoutCreate(&adesc, CUDA_R_16BF, mm, kk, mm);
                cublasLtMatrixLayoutCreate(&bdesc, CUDA_R_16BF, kk, nn, kk);
                cublasLtMatrixLayoutCreate(&cdesc, CUDA_R_16BF, mm, nn, mm);

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

                if (returnedResults == 0)
                {
                    cout << "WARNING: no algorithm found for "
                         << mm << "x" << nn << "x" << kk << endl;
                    continue;
                }

                for (int itr = 0; itr < 10; itr++)
                {
                    cublasLtMatmul(ltHandle, opDesc, &alpha, d_a, adesc, d_b, bdesc, &beta, d_c, cdesc, d_c, cdesc, &heuristicResult.algo, workspace, workspaceSize, 0);
                }
                cudaDeviceSynchronize(); // GPUの完了を待つ

                // 測定開始
                cudaEventRecord(start);
                for (int itr = 0; itr < 100; itr++)
                {
                    cublasLtMatmul(ltHandle, opDesc, &alpha, d_a, adesc, d_b, bdesc, &beta, d_c, cdesc, d_c, cdesc, &heuristicResult.algo, workspace, workspaceSize, 0);
                }
                cudaEventRecord(stop);
                cudaEventSynchronize(stop); // GPUの完了を待つ
                cudaEventElapsedTime(&sumtime, start, stop);
                
                // 性能計算 (2 * M * N * K / 時間)
                double ops = 2.0 * (long long)mm * (long long)nn * (long long)kk;
                double tops = (ops / (sumtime / 100.0 / 1000.0)) / 1e12;
                cout << "m" << mm << ", n" << nn << ", k" << kk << endl;
                string text = to_string(mm) + ", " + to_string(nn) + ", " + to_string(kk) +
                              ", " + to_string(sumtime) + ", " + to_string(tops);
                S.push_back(text);
                // 掃除
                cublasLtMatrixLayoutDestroy(adesc);
                cublasLtMatrixLayoutDestroy(bdesc);
                cublasLtMatrixLayoutDestroy(cdesc);
                cublasLtMatmulPreferenceDestroy(preference);
            }
        }
    }
    // cout << "Matrix Size: " << m << "x" << n << "x" << k << endl;
    // cout << "Execution Time: " << sumtime << " ms" << endl;
    // cout << "Performance: " << tops << " TOPS (Int8)" << endl;

    // 後片付け
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasLtMatmulDescDestroy(opDesc)
    cublasLtDestroy(ltHandle);
    cudaFree(workspace);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    write_file(S);
    //理論値95%と非常に良い結果。
    //おそらく① 出力がINT32（4byte）なのでメモリ書き込みが重い
    //INT8: 入力1byte × 2 + 出力4byte = 非効率
    //BF16: 入力2byte × 2 + 出力2byte = 均一
    return 0;
}