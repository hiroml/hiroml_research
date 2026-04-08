#include <bits/stdc++.h>
#include <cublas_v2.h>

using namespace std;
#define TILE_SIZE 32

__global__ void sgemm_tiled(int M, int N, int K,
                             float alpha,
                             const float *A,
                             const float *B,
                             float beta,
                             float *C) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float acc = 0.0f;
    //k回のループ
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        //tはタイル番号
        //ひとまず足して掛け算したC（縮小版）を作って、それを後でまとめて足し合わせる
        //As:Aのタイル行列
        //Bs:Bのタイル行列
        // ここを実装する
        // 1. As[threadIdx.y][threadIdx.x] に A の値をロード
        As[threadIdx.x][threadIdx.y] = A[threadIdx.y*t+threadIdx.x];
        // 2. Bs[threadIdx.y][threadIdx.x] に B の値をロード
        Bs[threadIdx.y][threadIdx.x] = B[threadIdx.y*t+threadIdx.x];
        // 3. __syncthreads()
        __syncthreads();
        // 4. タイル内で acc に積和
        acc += As[col * K][t] * Bs[row * N][t];
        // 5. __syncthreads()（次のタイルロード前）
        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = alpha * acc + beta * C[row * N + col];
}

__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C)
{
    // 各スレッドでの計算？
    // おそらくx,yの位置が一次元になっている。
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    // `if` condition is necessary for when M or N aren't multiples of 32.
    if (x < M && y < N)
    {
        float tmp = 0.0;
        for (int i = 0; i < K; ++i)
        {
            tmp += A[x * K + i] * B[i * N + y];
        }
        // C = α*(A@B)+β*C
        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}



int main(void)
{

    int M = 4096, K = 4096, N = 4096;
    float alpha = 1.0, beta = 0.0;
    // float *A;
    // float *B;
    // float *C;

    // A = (float *)malloc(M * N * sizeof(float));
    // B = (float *)malloc(N * K * sizeof(float));
    // C = (float *)malloc(M * K * sizeof(float));
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    // vectorでの渡し方わからなかった
    //  vector<float A(N*M);
    //  vector<float> B(M*K);
    //  vector<float> C(N*K);
    // numblocks
    dim3 gridDim(M / 32, N / 32, 1);
    // threadsperblock
    dim3 blockDim(32, 32, 1);

    // 計測
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float nvms;
    cudaEventElapsedTime(&nvms, start, stop);

    // cublas

    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaEventRecord(start);
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                M, N, K,
                &alpha,
                d_B, M, // ← cuBLASはcolumn-major
                d_A, K,
                &beta,
                d_C, M);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float blms;
    cudaEventElapsedTime(&blms, start, stop);

    double nvGFLOPS = (2.0 * M * N * K) / (nvms/1000.0 * 1e9);
    double blGFLOPS = (2.0 * M * N * K) / (blms/1000.0 * 1e9);
    cout << "nvGFLOPS:  " << nvGFLOPS << endl;
    cout << "blGFLOPS:  " << blGFLOPS << endl;
    cout << "達成率" << nvGFLOPS/blGFLOPS*100 << "%" << endl;

    return 0;
}