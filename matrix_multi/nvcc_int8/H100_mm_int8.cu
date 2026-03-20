#include <iostream>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <fstream>
#include <bits/stdc++.h>
#define rep(i, n) for (int i = 3; i < (n); ++i)

using namespace std;

// エラーチェック関数（省略せずに記述）
void checkCuda(cudaError_t err)
{
    if (err != cudaSuccess)
    {
        cerr << "CUDA Error: " << cudaGetErrorString(err) << endl;
        exit(1);
    }
}
void checkCublas(cublasStatus_t stat)
{
    if (stat != CUBLAS_STATUS_SUCCESS)
    {
        cerr << "cuBLAS Error: " << stat << endl;
        exit(1);
    }
}

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
    size_t workspaceSize = 32 * 1024 * 1024;
    void* d_workspace;
    checkCuda(cudaMalloc(&d_workspace, workspaceSize));
    vector<string> S;
    int n = 17;
    rep(i, n)
    {
        rep(j, n)
        {
            rep(l, n)
            {

                int m = pow(2, i), n = pow(2, j), k = pow(2, l); // サイズを大きくすると性能が出やすい
                cublasLtHandle_t ltHandle;
                checkCublas(cublasLtCreate(&ltHandle));

                // メモリ確保 (必要量を確認してから確保する)
                int8_t *d_a, *d_b;
                int32_t *d_c;
                size_t freeMem = 0, totalMem = 0;
                checkCuda(cudaMemGetInfo(&freeMem, &totalMem));
                size_t bytes_a = (size_t)m * (size_t)k * sizeof(int8_t);
                size_t bytes_b = (size_t)k * (size_t)n * sizeof(int8_t);
                size_t bytes_c = (size_t)m * (size_t)n * sizeof(int32_t);
                size_t required = bytes_a + bytes_b + bytes_c;
                // 保守マージンを10MB確保しておく
                const size_t margin = 10ull * 1024ull * 1024ull;
                if (required + margin > freeMem)
                {
                    cerr << "Skipping m=" << m << " n=" << n << " k=" << k
                         << " required=" << required << " free=" << freeMem << "\n";
                    continue;
                }
                checkCuda(cudaMalloc(&d_a, bytes_a));
                checkCuda(cudaMalloc(&d_b, bytes_b));
                checkCuda(cudaMalloc(&d_c, bytes_c));

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
                float sumtime = 0;
                // 本計測開始
                for (int itr = 0; itr < 100; itr++)
                {
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
                double tops = (ops / (sumtime / 1000.0)) * 100 / 1e12;

                // cout << "Matrix Size: " << m << "x" << n << "x" << k << endl;
                // cout << "Execution Time: " << sumtime << " ms" << endl;
                // cout << "Performance: " << tops << " TOPS (Int8)" << endl;
                cout <<"m"<< m <<", n"<< n<<", k" << k<< endl;
                string text = to_string(m) + ", " + to_string(n) + ", " + to_string(k) + 
              ", " + to_string(sumtime) + ", " + to_string(tops);
                S.push_back(text);
                // 後片付け
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
                cudaFree(d_a);
                cudaFree(d_b);
                cudaFree(d_c);
            }
        }
    }
    write_file(S);
    return 0;
}