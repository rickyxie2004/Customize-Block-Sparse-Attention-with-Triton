import triton
import triton.language as tl

from fa2_custom_mask.utils import is_hip, keep

# We don't run auto-tuning every time to keep the tutorial fast. Keeping
# the code below and commenting out the equivalent parameters is convenient for
# re-tuning.
configs = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w) \
    # for BM in [64, 128]\
    # for BN in [32, 64]\
    for BM in [64]\
    for BN in [64]\
    for s in ([1] if is_hip() else [3, 4, 7])\
    for w in [4, 8]\
]

@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,  #
    K_block_ptr,
    V_block_ptr,  #
    mask_block_ptr, # TODO: make sure it's added
    start_m, #
    qk_scale,  #
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,  #
    STAGE: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,  #
    N_CTX: tl.constexpr,
    fp8_v: tl.constexpr,
    USE_MASK: tl.constexpr,
):
    """

    """
    # range of values handled by this stage
    # if STAGE == 1:
    #     lo, hi = 0, start_m * BLOCK_M
    # elif STAGE == 2:
    #     lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
    #     lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    # else:
        # lo, hi = 0, N_CTX
    lo, hi = 0, N_CTX

    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    
    if USE_MASK:
        # TODO: advance mask pointer along N dim
        mask_block_ptr = tl.advance(mask_block_ptr, (0, lo))

    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(K_block_ptr)
        qk = tl.dot(q, k)
        if USE_MASK:
            # mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            # TODO: replace mask!
            mask_ = tl.load(mask_block_ptr)
            # for block mask
            mask_ = mask_.to(tl.int1) 
            qk = qk * qk_scale + tl.where(mask_, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        v = tl.load(V_block_ptr)
        if fp8_v:
            p = p.to(tl.float8e5)
        else:
            p = p.to(tl.float16)
        acc = tl.dot(p, v, acc)
        # update m_i and l_i
        m_i = m_ij

        # TODO: is this wrong? no, just stride
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))

        # TODO: update mask pointer offset
        if USE_MASK:
            # mask_block_ptr = tl.advance(mask_block_ptr, (0, BLOCK_N))
            mask_block_ptr = tl.advance(mask_block_ptr, (0, 1))
            
    return acc, l_i, m_i


@triton.autotune(list(filter(keep, configs)), key=["N_CTX", "HEAD_DIM"])
@triton.jit
def _attn_fwd(Q, K, V, groups_ptr, nlist_ptr, sm_scale, M, Out,  #
            stride_qz, stride_qh, stride_qm, stride_qk,  #
            stride_kz, stride_kh, stride_kn, stride_kk,  #
            stride_vz, stride_vh, stride_vk, stride_vn,  #
            stride_oz, stride_oh, stride_om, stride_on,  #
            B, H, N_CTX,  #
            HEAD_DIM: tl.constexpr,  #
            BLOCK_M: tl.constexpr,  #
            BLOCK_N: tl.constexpr,  #
              ):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    
    g = tl.program_id(0)
    base_groups = groups_ptr + g * 5
    b = tl.load(base_groups + 0).to(tl.int32)
    h = tl.load(base_groups + 1).to(tl.int32)
    m_block = tl.load(base_groups + 2).to(tl.int32)
    start_idx = tl.load(base_groups + 3).to(tl.int32)
    length = tl.load(base_groups + 4).to(tl.int32)

    qvk_offset = b * stride_qz + h * stride_qh
    q_base = Q + qvk_offset
    k_base = K + qvk_offset
    v_base = V + qvk_offset
    o_base = Out + b * stride_oz + h * stride_oh

    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=q_base,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(m_block * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    q = tl.load(Q_block_ptr)
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    qk_scale = sm_scale * 1.44269504

    for idx in range(length):
        n_index = tl.load(nlist_ptr + (start_idx + idx))
        n_block = n_index.to(tl.int32)
        K_block_ptr = tl.make_block_ptr(
            base=k_base,
            shape=(HEAD_DIM, N_CTX),
            strides=(stride_kk, stride_kn),
            offsets=(0, n_block * BLOCK_N),
            block_shape=(HEAD_DIM, BLOCK_N),
            order=(0, 1),
        )
        v_order: tl.constexpr = (0, 1) if V.dtype.element_ty == tl.float8e5 else (1, 0)
        fp8_v = V.dtype.element_ty == tl.float8e5
        V_block_ptr = tl.make_block_ptr(
            base=v_base,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_vk, stride_vn),
            offsets=(n_block * BLOCK_N, 0),
            block_shape=(BLOCK_N, HEAD_DIM),
            order=v_order,
        )

        k = tl.load(K_block_ptr)
        qk = tl.dot(q, k)
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        v = tl.load(V_block_ptr)
        
        # if fp8_v:
        #     p = p.to(tl.float8e5)
        # else:
        p = p.to(v.dtype)
        acc = tl.dot(p, v, acc)
        m_i = m_ij
    
    acc = acc / l_i[:, None]
    off_hz = b * H + h
    m_ptrs = M + off_hz * N_CTX + (m_block * BLOCK_M + tl.arange(0, BLOCK_M))
    tl.store(m_ptrs, m_i)

    O_block_ptr = tl.make_block_ptr(
        base=o_base,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(m_block * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))

