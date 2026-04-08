# 実行ファイル名を a.out に設定
TARGET  = a.out
SRCS    = sgemm_tiled.cu
OBJS    = sgemm_tiled.o
CC      = nvcc

# ライブラリ指定
LIBS    = -lcublas

# デフォルトターゲット
all: clean $(TARGET)

# リンク工程 (nvcc native_gemm.o -lcublas)
$(TARGET): $(OBJS)
	$(CC) -o $@ $^ $(LIBS)

# コンパイル工程 (nvcc -c native_gemm.cu)
$(OBJS): $(SRCS)
	$(CC) -c $< -o $@

# クリーンアップ
clean:
	-rm -f $(OBJS) $(TARGET)

.PHONY: all clean