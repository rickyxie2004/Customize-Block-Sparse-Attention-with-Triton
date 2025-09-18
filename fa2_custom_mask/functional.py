import torch
from .permute import reorder_kernel, inv_reorder_kernel
import triton

def apply_head_wise_reorder(tensor, permute_orders, F, H, W):
    BS, N_head, N_token, D = tensor.shape
    assert N_token == F * H * W
    output = torch.empty_like(tensor)
    BLOCK = 1024
    grid = lambda META: (triton.cdiv(BS * N_head * N_token, BLOCK),)

    reorder_kernel[grid](
        tensor, output,
        BS, N_head, N_token, D,
        F, H, W,
        permute_orders,
        tensor.stride(0), tensor.stride(1), tensor.stride(2), tensor.stride(3),
        BLOCK=BLOCK,
    )
    return output

def permute_qkv_triton(query, key, value, permute_plan, i_block, F=13, H=30, W=45, n_text_tokens=226):
    permutations = torch.tensor([
        [0, 1, 2], [0, 2, 1], [1, 2, 0],
        [1, 0, 2], [2, 1, 0], [2, 0, 1],
    ], dtype=torch.int32, device=query.device)

    idx = permute_plan['permute'][i_block]  # shape [N_head]
    permute_orders = permutations[idx]      # shape [N_head, 3]

    query_image = query[:, :, n_text_tokens:, :]
    key_image = key[:, :, n_text_tokens:, :]
    value_image = value[:, :, n_text_tokens:, :]

    query_out = apply_head_wise_reorder(query_image.contiguous(), permute_orders, F, H, W)
    key_out = apply_head_wise_reorder(key_image.contiguous(), permute_orders, F, H, W)
    value_out = apply_head_wise_reorder(value_image.contiguous(), permute_orders, F, H, W)

    query[:, :, n_text_tokens:, :] = query_out
    key[:, :, n_text_tokens:, :] = key_out
    value[:, :, n_text_tokens:, :] = value_out

    return query, key, value


def apply_inv_reorder(tensor, permute_orders, inv_permute_orders, F, H, W):
    BS, N_head, N_token, D = tensor.shape
    assert N_token == F * H * W
    output = torch.empty_like(tensor)
    BLOCK = 1024
    grid = lambda META: (triton.cdiv(BS * N_head * N_token, BLOCK),)

    inv_reorder_kernel[grid](
        tensor, output,
        BS, N_head, N_token, D,
        F, H, W,
        permute_orders,
        inv_permute_orders,
        tensor.stride(0), tensor.stride(1), tensor.stride(2), tensor.stride(3),
        BLOCK=BLOCK,
    )
    return output

def inv_permute_attn_triton(attn_out, permute_plan, i_block, F=13, H=30, W=45, n_text_tokens=226):
    permutations = torch.tensor([
        [0, 1, 2], [0, 2, 1], [1, 2, 0],
        [1, 0, 2], [2, 1, 0], [2, 0, 1],
    ], dtype=torch.int32, device=attn_out.device)
    permutations_inv = torch.tensor([
        [0, 1, 2],  # 0: FHW
        [0, 2, 1],  # 1: FWH
        [2, 0, 1],  # 2: HWF
        [1, 0, 2],  # 3: HFW
        [2, 1, 0],  # 4: WHF
        [1, 2, 0],  # 5: WFH
    ], dtype=torch.int32, device=attn_out.device)
    BS, N_head, N_token, D = attn_out.shape
    idx = permute_plan['permute'][i_block]  # shape [N_head]
    permute_orders = permutations[idx]      # shape [N_head, 3]
    inv_permute_orders = permutations_inv[idx]
    attn_image = attn_out[:,:,n_text_tokens:,:]
    attn_image_out = apply_inv_reorder(attn_image.contiguous(), permute_orders, inv_permute_orders, F, H, W)
    attn_out[:, :, n_text_tokens:, :] = attn_image_out
    return attn_out
    

