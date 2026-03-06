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
    size = 2048 
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
    
    n_iter = 100
    start_time = time.perf_counter()
    
    for i in range(n_iter):
        res = matmul_basic(mat_a, mat_b)
        # 非同期実行を待機させるのが計測のコツ
        res.block_until_ready()
        
    end_time = time.perf_counter()
    sum_time= (end_time-start_time)
    avg_time = (end_time - start_time) / n_iter
    print(f"実行時間{sum_time}")
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

