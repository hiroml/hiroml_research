# Cudaを扱うにあたり必要な関数のメモ
cublasも含む

# cudacpp

## cudamalloc
```
cudaError_t cudaMalloc(void** devPtr, size_t size);

```

引数（Input/Output）
devPtr (出力): 割り当てられたメモリへのポインタを格納するアドレスです。void** 型であるため、通常は (void**)&d_ptr のように渡します。

size (入力): 割り当てたいメモリのサイズをバイト単位で指定します。

戻り値（Return Value）
cudaError_t: 実行結果を返します。成功した場合は cudaSuccess を返し、失敗した場合はエラーコード（例：cudaErrorMemoryAllocation など）を返します。


## cudaEventCreate
cudaEventCreate は、GPUの実行ストリーム内に「チェックポイント（目印）」を打つためのイベントオブジェクトを作成します。

```
cudaError_t cudaEventCreate(cudaEvent_t* event);
```

引数（Output）
event:
新しく作成されたイベントオブジェクトのハンドルを格納するポインタです。

戻り値
cudaError_t:
成功した場合は cudaSuccess を返します。

## cudaEventDestroy
cudaEventDestroy は、指定されたイベントオブジェクトを破棄し、それに関連付けられていたGPUおよびホスト側のリソースを解放します。
```
cudaEventDestroy(start);
```

引数
event:
破棄したいイベントのハンドルを渡します。

戻り値
cudaError_t:
成功した場合は cudaSuccess を返します。無効なイベントハンドルを渡した場合は cudaErrorInvalidResourceHandle などのエラーが返ることがあります。


## cudaFree
cudaFree は、デバイス上のグローバルメモリ割り当てを解除し、その領域を再利用可能にします。

```
cudaFree(d_a)
```
引数
devPtr:
解放したいデバイスメモリへのポインタです。以前に cudaMalloc や cudaMallocPitch などで取得したアドレスを渡します。

注記: NULL ポインタ（0）を渡した場合、cudaFree は単に何もせず cudaSuccess を返します。これは標準Cの free() と同じ挙動です。

戻り値
cudaError_t:
通常は cudaSuccess を返します。


# cublas
## cublasLtMatmulDescCreate

この関数は、行列演算の計算精度、データ型、および演算の「数学的性質」を定義するオブジェクトを作成します。
```
cublasStatus_t cublasLtMatmulDescCreate(
    cublasLtMatmulDesc_t* matmulDesc,
    cublasComputeType_t computeType,
    cudaDataType_t scaleType
);
```

引数（Input/Output）matmulDesc (出力):作成された記述子を格納するハンドルへのポインタです。このハンドルを通じて、後続の演算設定（転置の有無など）を行います。
computeType (入力):計算の「精度」を指定します。例: CUBLAS_COMPUTE_32F (FP32での計算), CUBLAS_COMPUTE_16F (FP16)最新の動向: CUDA 11/12以降では、TF32やFP8（CUBLAS_COMPUTE_8F_E4M3 など）のサポートが追加されており、計算効率を飛躍的に高めることが可能です。
scaleType (入力):行列乗算の結果に乗じるスカラー値（$\alpha, \beta$）のデータ型を指定します。通常は computeType と一致させますが、混合精度演算では異なる場合があります。戻り値cublasStatus_t: 成功時は CUBLAS_STATUS_SUCCESS。メモリ不足時は CUBLAS_STATUS_ALLOC_FAILED が返されます。

cublasLt を利用する最大のメリットは、「エピローグ（Epilogue）」の融合にあります。
通常、行列乗算の後に「バイアス加算」や「活性化関数（ReLUなど）」を適用する場合、別途カーネルを起動する必要がありますが、cublasLtMatmulDescSetAttribute でこれらを指定すれば、1つのカーネル内で完結させることができ、メモリ帯域の節約につながります。

## cublasLtMatmul

```
    cublasLtMatmul(ltHandle, opDesc, &alpha, d_a, adesc, d_b, bdesc, &beta, d_c, cdesc, d_c, cdesc, NULL, NULL, 0, 0);

```

この関数は、記述子（Descriptor）に基づいて、計算の実行を指示します。
主な引数の役割lightHandle: cuBLASLtのコンテキストハンドル。computeDesc: cublasLtMatmulDescCreate で作成した、演算の定義（精度やエピローグ）。A, B, C: 入力行列。$D = \alpha(A \times B) + \beta C$ の計算を行います。D: 出力行列。CとDを同じアドレスにすれば、インプレース更新が可能です。adesc, bdesc, cdesc, ddesc: 各行列のレイアウト（次元、ストライド、データ型）を定義する記述子。algo: 極めて重要。 どの実装（カーネル）を使用するかを指定します。通常は cublasLtMatmulAlgoGetHeuristic で取得した最適なアルゴリズムを渡します。workspace: 演算中に使用する一時的なGPUメモリ領域。

# その他
上限メモリの計算方法
必要メモリ = int8*m*k + int8*k*n + int32*m*n
           = m*k + k*n + 4*m*n  (bytes)

m=n=k=N の正方行列の場合:
= N^2 + N^2 + 4*N^2 = 6*N^2
6 * N^2 <= 12GB = 12 * 1024^3
N^2 <= 2 * 1024^3 = 2,147,483,648
N <= 46,341  → 実用上は N = 32768 (2^15) が限界

N=32768 の場合:
d_a: 1GB
d_b: 1GB  
d_c: 4GB
合計: 6GB ← 12GBに収まる ✅

N=65536の場合
6*N^2=
= 4,294,967,296 + 4,294,967,296 +4,294,967,296*4
=25,769,803,776(byte)
=24GBくらい

80GBあった場合
6*n^2 = 80gb
n^2 <= 85,899,345,920 / 6 = 14,316,557,653
N <= √14,316,557,653 ≈ 119,652