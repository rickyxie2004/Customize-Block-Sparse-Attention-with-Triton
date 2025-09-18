import torch
import triton
import triton.language as tl

@triton.jit
def reorder_kernel(
    input_ptr, output_ptr,
    bs, n_heads, n_tokens, dim,
    F, H, W,
    permute_orders_ptr,
    stride_b, stride_h, stride_t, stride_d,
    BLOCK: tl.constexpr,
):
    # [BS, N_head, F, H, W, D]
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    total = bs * n_heads * n_tokens
    mask = off < total

    # 还原 (b, h, t) 索引
    b = off // (n_heads * n_tokens)
    off = off % (n_heads * n_tokens)
    h = off // n_tokens
    t = off % n_tokens

    # 还原为原始的 (f, h_, w) 坐标
    fw = H * W
    f = t // fw
    t_ = t % fw
    h_ = t_ // W
    w = t_ % W

    # 加载该 head 的 permute 顺序
    p0 = tl.load(permute_orders_ptr + h * 3 + 0)
    p1 = tl.load(permute_orders_ptr + h * 3 + 1)
    p2 = tl.load(permute_orders_ptr + h * 3 + 2)

    # 原始三维坐标
    coords = [f, h_, w]
    dims = [F, H, W]

    # 根据 permute 顺序计算新的 (i, j, k)
     # 原始坐标：f, h_, w
    # permute: p0, p1, p2 in {0,1,2} 表示 f/h_/w

    c0 = f * (p0 == 0) + h_ * (p0 == 1) + w * (p0 == 2)
    c1 = f * (p1 == 0) + h_ * (p1 == 1) + w * (p1 == 2)
    c2 = f * (p2 == 0) + h_ * (p2 == 1) + w * (p2 == 2)

    d1 = F * (p1 == 0) + H * (p1 == 1) + W * (p1 == 2)
    d2 = F * (p2 == 0) + H * (p2 == 1) + W * (p2 == 2)

    new_t = c0 * (d1 * d2) + c1 * d2 + c2

    for d in range(0, dim):
        in_idx = b * stride_b + h * stride_h + t * stride_t + d * stride_d
        out_idx = b * stride_b + h * stride_h + new_t * stride_t + d * stride_d
        val = tl.load(input_ptr + in_idx, mask=mask)
        tl.store(output_ptr + out_idx, val, mask=mask)

@triton.jit
def inv_reorder_kernel(
    input_ptr, output_ptr,
    bs, n_heads, n_tokens, dim,
    F, H, W,
    permute_orders_ptr,
    inv_permute_orders_ptr,
    stride_b, stride_h, stride_t, stride_d,
    BLOCK: tl.constexpr,
):
    # [BS, N_head, F, H, W, D]
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    total = bs * n_heads * n_tokens
    mask = off < total

    # 还原 (b, h, t) 索引
    b = off // (n_heads * n_tokens)
    off = off % (n_heads * n_tokens)
    h = off // n_tokens
    t = off % n_tokens

    fp = tl.load(permute_orders_ptr + h *3 + 0)
    hp = tl.load(permute_orders_ptr + h *3 + 1)
    wp = tl.load(permute_orders_ptr + h *3 + 2)

    x1 = F * (fp==0) + H * (fp==1) + W * (fp==2)
    x2 = F * (hp==0) + H * (hp==1) + W * (hp==2)
    x3 = F * (wp==0) + H * (wp==1) + W * (wp==2)

    F = x1
    H = x2
    W = x3

    # 还原为原始的 (f, h_, w) 坐标
    fw = H * W
    f = t // fw
    t_ = t % fw
    h_ = t_ // W
    w = t_ % W

    # 加载该 head 的 permute 顺序
    p0 = tl.load(inv_permute_orders_ptr + h * 3 + 0)
    p1 = tl.load(inv_permute_orders_ptr + h * 3 + 1)
    p2 = tl.load(inv_permute_orders_ptr + h * 3 + 2)

    # 原始三维坐标
    coords = [f, h_, w]
    dims = [F, H, W]

    # 根据 permute 顺序计算新的 (i, j, k)
     # 原始坐标：f, h_, w
    # permute: p0, p1, p2 in {0,1,2} 表示 f/h_/w

    c0 = f * (p0 == 0) + h_ * (p0 == 1) + w * (p0 == 2)
    c1 = f * (p1 == 0) + h_ * (p1 == 1) + w * (p1 == 2)
    c2 = f * (p2 == 0) + h_ * (p2 == 1) + w * (p2 == 2)

    d1 = F * (p1 == 0) + H * (p1 == 1) + W * (p1 == 2)
    d2 = F * (p2 == 0) + H * (p2 == 1) + W * (p2 == 2)

    new_t = c0 * (d1 * d2) + c1 * d2 + c2

    for d in range(0, dim):
        in_idx = b * stride_b + h * stride_h + t * stride_t + d * stride_d
        out_idx = b * stride_b + h * stride_h + new_t * stride_t + d * stride_d
        val = tl.load(input_ptr + in_idx, mask=mask)
        tl.store(output_ptr + out_idx, val, mask=mask)
