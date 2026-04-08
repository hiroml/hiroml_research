# TPUの概要
- 大規模行列演算向け
- 乗算演算と累積演算
- 乗算アキュムレータが大量
- シストリック　アレイ　アーキテクチャの採用
- V6e,7xは256*256,それ以前は128×128


# シントリックアレイアーキテクチャ
- 行列演算目的
- 乗算加算器が網目状に並んでいる。
- 掛け算をして、加算で流れていくのでメモリ書き出しが必要なく高速
- 1クロックごとに計算結果が隣に流れていく。
- メモリのかき戻しが必要ないのでいい感じ

# google cloud TPUの環境構築
https://docs.cloud.google.com/tpu/docs/setup-gcp-account?hl=ja

## 1. google CLIのインストール
以下を参考にする
https://docs.cloud.google.com/sdk/docs/install-sdk?hl=ja

### ダウンロード
```CLIinstall
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz
```
### 解凍
```
tar -xf google-cloud-cli-linux-x86_64.tar.gz
```
### スクリプトの実行(install)
```
./google-cloud-sdk/install.sh
```

### 初期化
初めに初期化が必要(認証も)いろいろ出てくるので回答する

```
gcloud init
```

myprojectを登録する。
使いたいプロジェクトのIDを確認し、登録する
```
gcloud config set project  project-6fed2baa-e4f8-40e0-a2d
```


## 2. google cloud TPUの予約
### 参考
https://docs.cloud.google.com/tpu/docs/setup-gcp-account?hl=ja
https://docs.cloud.google.com/tpu/docs/tpu7x?hl=ja#configurations

今回は以下の要件で利用
- Zone: asia-northeast1-b
- ACCELERATOR_TYPE: v6e-1
- RUNTIME_VERSION: v2-alpha-tpuv6e

### 1. TPU APIの利用承認
使いたいプロジェクトのgoogle cloud の cloud TPU APIを利用可能にする
```
gcloud services enable tpu.googleapis.com
```
またはwebからできる
![alt text](image.png)

### 2. TPUの利用予約
初めに使いたいTPUとzoneを決める
https://docs.cloud.google.com/tpu/docs/regions-zones?hl=ja
今回はv6e-1,asia-northeast1-b(Tokyo)

####  使えるアクセラレータタイプを探す
アクセラレータはvmのチップの種類(数など)
TPUバージョンのページから確認できる
今回はお試しでv6e-1(テスト向けチップ1つ)
https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm?hl=ja#versions

```
gcloud compute tpus tpu-vm accelerator-types list --zone=asia-northeast1-b
```


#### 使えるソフトウェアバージョンの確認
使うzoneとTPUによってソフトウェアバージョンが異なる確認は以下
https://docs.cloud.google.com/tpu/docs/runtimes?hl=ja
```
gcloud compute tpus tpu-vm versions list --zone=asia-northeast1-b
```

### TPUの予約・接続
使いやすいように変数を登録する
```
export PROJECT_ID=project-6fed2baa-e4f8-40e0-a2d
export TPU_NAME=v6e-tokyo
export ZONE=us-east5-a
export ACCELERATOR_TYPE=v6e-1
export RUNTIME_VERSION=v2-alpha-tpuv6e
```

```
export PROJECT_ID=project-6fed2baa-e4f8-40e0-a2d
export TPU_NAME=v5e-us
export ZONE=us-west1-c
export ACCELERATOR_TYPE=v5litepod-8
export RUNTIME_VERSION=v2-alpha-tpuv5-lite
```


以下で予約できる。
```
gcloud compute tpus tpu-vm create $TPU_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    --accelerator-type=$ACCELERATOR_TYPE \
    --version=$RUNTIME_VERSION
    --spot
```


## 4. ssh接続
以下コマンドでssh接続を行う。
最初はパスフレーズの登録が必要

```
gcloud compute tpus tpu-vm ssh $TPU_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE
```

## 5. 必要ライブラリのインストール
TPUの利用にはJAXというライブラリ(python)を用いてコードをstableHLOへの変換が必要
- JAX: 簡単にStableHLOへの変換ができるやつ
- stableHLO: TPUがそれぞれ異なる命令セットだと困るので、それぞれに当てはめるための中間言語 OpenXLAというプロジェクトの一部らしい

以下を実行
```
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

pythonでjaxバージョンの確認
```
python3 
```
```
import jax
jax.device_count()
```
結果
```
1
```

### クリーンアップ
通常予約状態だと接続していなくても課金が発生してしまう。使わない場合は削除する


切断
```
exit
```
削除
```
gcloud compute tpus tpu-vm delete $TPU_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE
```
確認
```
gcloud compute tpus tpu-vm list \
    --zone=$ZONE
```

# google cloud TPUでのコーディング

簡単なdgemm
```
import os
import time
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils

def main():
    # 1. デバイス（TPUコア）の確認
    devices = jax.devices()
    num_devices = len(devices)
    print(f"利用可能なデバイス数: {num_devices}")
    print(f"デバイス一覧: {devices}")

    # 2. データの準備 (bfloat16を使うとTPUの性能が最大化されます)
    size = 4096 
    key = jax.random.PRNGKey(42)
    
    print(f"行列サイズ: {size}x{size} を作成中...")
    mat_a = jax.random.normal(key, (size, size), dtype=jnp.bfloat16)
    mat_b = jax.random.normal(key, (size, size), dtype=jnp.bfloat16)

    # 3. 行列積の関数定義とJITコンパイル
    @jax.jit
    def matmul_basic(a, b):
        return jnp.matmul(a, b)

    # --- 実行時間の計測セクション ---
    print("計測開始...")
    
    # ウォームアップ（初回はコンパイルが入るため計測から除外）
    _ = matmul_basic(mat_a, mat_b).block_until_ready()
    
    n_iter = 10
    start_time = time.perf_counter()
    
    for i in range(n_iter):
        res = matmul_basic(mat_a, mat_b)
        # 非同期実行を待機させるのが計測のコツ
        res.block_until_ready()
        
    end_time = time.perf_counter()
    avg_time = (end_time - start_time) / n_iter
    print(f"平均実行時間 ({n_iter}回): {avg_time:.6f} 秒")

    # --- 並列化（Sharding）のデモセクション ---
    print("\n並列化の準備 (Sharding) を実行...")
    
    # 1次元のデバイスメッシュを作成 (全コアを並列利用)
    mesh = Mesh(mesh_utils.create_device_mesh((num_devices,)), axis_names=('x',))
    
    # 行列Aを行方向に分割、行列Bは全デバイスに複製
    sharding_a = NamedSharding(mesh, P('x', None))
    sharded_a = jax.device_put(mat_a, sharding_a)
    
    # この sharded_a を matmul_basic に渡すだけで、
    # JAXが自動的にマルチコア並列計算（SPMD）に切り替えます。
    res_parallel = matmul_basic(sharded_a, mat_b)
    res_parallel.block_until_ready()
    
    print("並列実行完了。")
    print(f"結果の配置情報: {res_parallel.sharding}")

if __name__ == "__main__":
    main()
```


# 今後の課題
- OpenXLAを利用したcppからの実行 
- 行列サイズを変えた時の実行時間比較
- 格子ボルツマン法とか試してみたい



# 単語
-