# FlashAttention2 with Masks 

Based on FA2 original triton kernel, addition to the code enables the FA2 to use block mask when inference. The block size is 64*64(so disable the auto adjustment of block size in FA2). The algorithm will avoid tl.dot when masked.
Original Triton code: [https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)

## Install
Create a Python environment (>=3.8) and install through pip:
```
pip install -e .
```

## Example Setup
The relevant libraries needed to use the custom-mask FlashAttention2 kernel are below:
```
pip install triton>=3.0.0
pip install torch
```

## For Viewing Benchmarking Results
```
python kernel-test.py
```


| 输入规模 [seq_len, dim] | 稀疏比 | FlashAttention2 custom mask 耗时 (秒) + 预处理mask | Naive matmul+softmax 耗时 (秒) | 加速倍数 (Naive / Flash) |
|-------------------------|--------|-------------------------------------|--------------------------------|--------------------------|
| [6400, 64]              | 0.1    | 0.0537                              | 0.6570                         | 12.23                   |
| [6400, 64]              | 1.0    | 0.0615                              | 0.6482                         | 10.54                   |
| [17536, 64]             | 1.0    | 0.2229                              | 0.6713                         | 3.01                    |
| [17536, 64]             | 0.1    | 0.1534                              | 0.6822                         | 4.45                    |
| [32000, 64]             | 0.1    | 0.2896                              | OOM                            | -                       |
| [32000, 64]             | 1.0    | 0.5324                              | OOM                            | -                       |

#### 单独考虑计算
N=17536

FlashAttention2 custom mask 耗时: 0.0096 秒

Naive matmul+softmax 耗时: 0.6306 秒

0.1稀疏

FlashAttention2 custom mask 耗时: 0.0013 秒
