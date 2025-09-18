import torch
import time
import math

# ========== 参数 ==========
BATCH = 1
HEADS = 8
SEQ_LEN = 17536
DIM = 64
DEVICE = "cuda"

# 随机生成 qkv
q = torch.randn(BATCH,  HEADS,SEQ_LEN, DIM, device=DEVICE, dtype=torch.float16, requires_grad=False)
k = torch.randn(BATCH,  HEADS,SEQ_LEN, DIM, device=DEVICE, dtype=torch.float16, requires_grad=False)
v = torch.randn(BATCH,  HEADS,SEQ_LEN, DIM, device=DEVICE, dtype=torch.float16, requires_grad=False)
p = 0.1
sparse = (torch.rand(BATCH, HEADS, SEQ_LEN//64, SEQ_LEN//64, device=DEVICE) < p).to(torch.bool)
expanded_sparse = sparse.repeat_interleave(64, dim=2).repeat_interleave(64, dim=3)
sm_scale = 1.0 / math.sqrt(DIM)

# ========== 方法1: FlashAttention2 with custom mask ==========
from fa2_custom_mask import flash_attention_custom_mask  # 你提到的函数
out_flash = flash_attention_custom_mask(q, k, v, sparse, sm_scale)  
torch.cuda.synchronize()
t0 = time.time()
out_flash = flash_attention_custom_mask(q, k, v, sparse, sm_scale)  
torch.cuda.synchronize()
t1 = time.time()
print(f"FlashAttention2 custom mask 耗时: {t1 - t0:.4f} 秒")

# ========== 方法2: 朴素实现 (矩阵乘+softmax) ==========
torch.cuda.synchronize()
t0 = time.time()

# q, k, v: [B, H, L, D]
# 注意力分数
attn = torch.matmul(q, k.transpose(-2, -1)) * sm_scale  # (B, H, L, L)
attn = attn.masked_fill(~expanded_sparse, float('-inf'))
attn = torch.softmax(attn, dim=-1)
out_naive = torch.matmul(attn, v)  # (B, H, L, D)

torch.cuda.synchronize()
t1 = time.time()
print(f"Naive matmul+softmax 耗时: {t1 - t0:.4f} 秒")

# ========== 正确性对比 ==========
print("结果shape对比:", out_flash.shape, out_naive.shape)

diff = (out_flash - out_naive).float()
print("L2误差:", torch.norm(diff).item())
print("最大绝对误差:", diff.abs().max().item())

