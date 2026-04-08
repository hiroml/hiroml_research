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
    results=[]
    results.append("n, m, k, sumtime, tops\n")
    sssize = 15
    n = sssize
    m= sssize
    
    for k in range(3,sssize):
        # 2. データの準備 (bfloat16を使うとTPUの性能が最大化されます)
        size_n = pow(2,m)
        size_m = pow(2,m)
        size_k = pow(2,k)
        key = jax.random.PRNGKey(42)
        
        mat_a = jax.random.randint(key, (size_n, size_m),minval=-128,maxval=127, dtype=jnp.int8)
        mat_b = jax.random.randint(key, (size_m, size_k),minval=-128,maxval=127, dtype=jnp.int8)

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
        ops = 2.0*size_m*size_n*size_k
        tops = (ops/sum_time/1000.0)*100/1e12
        s = str(size_n) + ", "+ str(size_m) +", "+ str(size_k) + ", "+str(sum_time)+", "+str(tops)+"\n"
        results.append(s)
    with open('./output_4.txt',mode='w') as f:
        for i in results:
            f.write(i)

if __name__ == "__main__":
    main()

