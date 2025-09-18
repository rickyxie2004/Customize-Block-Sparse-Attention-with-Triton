import torch
import triton

from fa2_custom_mask.fa2_fwd import _attn_fwd
from fa2_custom_mask.fa2_bwd import _attn_bwd_preprocess, _attn_bwd
from fa2_custom_mask.utils import is_hip

# -------------------------
# Reminder: build groups & n_list from block mask
# mask: [B, H, N_block, N_block], bool
# returns:
#   groups: int32 tensor shape [G, 5] columns = (b, h, m_block, start_idx, length)
#   n_list: int32 tensor shape [total_active_n]
# -------------------------
def build_groups_from_block_mask(mask: torch.Tensor):
    """
    mask: bool tensor shape [B, H, N_block, N_block]
    returns groups (G,5) and n_list (total_active,)
    """
    assert mask.dtype == torch.bool
    B, H, N_block, _ = mask.shape
    groups = []
    n_list = []
    for b in range(B):
        for h in range(H):
            # iterate m_block
            for m in range(N_block):
                active_ns = torch.nonzero(mask[b, h, m], as_tuple=False).squeeze(1)
                if active_ns.numel() == 0:
                    continue
                start = len(n_list)
                n_list.extend(active_ns.tolist())
                groups.append([b, h, m, start, active_ns.numel()])
    if len(groups) == 0:
        return None, None
    groups_t = torch.tensor(groups, dtype=torch.int32, device=mask.device)
    nlist_t = torch.tensor(n_list, dtype=torch.int32, device=mask.device)
    return groups_t.contiguous(), nlist_t.contiguous()

class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, mask=None, sm_scale=1.3):
        # shape constraints
        B, H, N_CTX, HEAD_DIM = q.shape[0], q.shape[1], q.shape[2], q.shape[3]
        N_block = N_CTX // 64
        if mask is not None:
            groups, nlist = build_groups_from_block_mask(mask)
            if groups is None:
                # all masked out
                return torch.zeros_like(q)
            
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        USE_MASK = mask is not None
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        o = torch.empty_like(q)

        # TODO: verify this means mask is not None
        stage = 3 if mask is not None else 2
        extra_kern_args = {}
        # Tuning for AMD target
        if is_hip():
            waves_per_eu = 3 if HEAD_DIM_K <= 64 else 2
            extra_kern_args = {"waves_per_eu": waves_per_eu, "allow_flush_denorm": True}

        grid = (groups.shape[0],) # all q blocks
        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)

        mask_stride_0 = (None if not USE_MASK else mask.stride(0)) # block
        mask_stride_1 = (None if not USE_MASK else mask.stride(1)) # head
        mask_stride_2 = (None if not USE_MASK else mask.stride(2)) # query
        mask_stride_3 = (None if not USE_MASK else mask.stride(3)) # key
        
        _attn_fwd[grid](
            q, k, v, groups, nlist, sm_scale, M, o,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            B, H, N_CTX,
            HEAD_DIM=HEAD_DIM,
            # BLOCK_M=64,
            # BLOCK_N=64,
            **extra_kern_args)

        ctx.save_for_backward(q, k, v, o, mask, M)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.USE_MASK = USE_MASK
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, mask, M = ctx.saved_tensors
        assert do.is_contiguous()
        assert q.stride() == k.stride() == v.stride() == o.stride() == do.stride()
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        BATCH, N_HEAD, N_CTX = q.shape[:3]
        PRE_BLOCK = 128
        NUM_WARPS, NUM_STAGES = 4, 5
        BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 128, 128, 32
        BLK_SLICE_FACTOR = 2
        RCP_LN2 = 1.4426950408889634  # = 1.0 / ln(2)
        arg_k = k
        arg_k = arg_k * (ctx.sm_scale * RCP_LN2)
        PRE_BLOCK = 128
        assert N_CTX % PRE_BLOCK == 0
        pre_grid = (N_CTX // PRE_BLOCK, BATCH * N_HEAD)
        delta = torch.empty_like(M)
        _attn_bwd_preprocess[pre_grid](
            o, do,  #
            delta,  #
            BATCH, N_HEAD, N_CTX,  #
            BLOCK_M=PRE_BLOCK, HEAD_DIM=ctx.HEAD_DIM  #
        )
        grid = (N_CTX // BLOCK_N1, 1, BATCH * N_HEAD)
        
        if ctx.USE_MASK:
            _attn_bwd[grid](
                q, arg_k, v, mask, ctx.sm_scale, do, dq, dk, dv,  #
                M, delta,  #
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
                mask.stride(0), mask.stride(1), mask.stride(2), mask.stride(3),  #
                N_HEAD, N_CTX,  #
                BLOCK_M1=BLOCK_M1, BLOCK_N1=BLOCK_N1,  #
                BLOCK_M2=BLOCK_M2, BLOCK_N2=BLOCK_N2,  #
                BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,  #
                HEAD_DIM=ctx.HEAD_DIM,
                USE_MASK=ctx.USE_MASK, #
                num_warps=NUM_WARPS,  #
                num_stages=NUM_STAGES,  #
            )
        else:
            _attn_bwd[grid](
                q, arg_k, v, None, ctx.sm_scale, do, dq, dk, dv,  #
                M, delta,  #
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
                None, None, None, None,  #
                N_HEAD, N_CTX,  #
                BLOCK_M1=BLOCK_M1, BLOCK_N1=BLOCK_N1,  #
                BLOCK_M2=BLOCK_M2, BLOCK_N2=BLOCK_N2,  #
                BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,  #
                HEAD_DIM=ctx.HEAD_DIM,
                USE_MASK=ctx.USE_MASK,#
                num_warps=NUM_WARPS,  #
                num_stages=NUM_STAGES  #
            )

        return dq, dk, dv, None, None
    
flash_attention_custom_mask = _attention.apply
