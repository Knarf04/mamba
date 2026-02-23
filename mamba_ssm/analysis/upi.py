import argparse

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers for saving / loading upi masks outside the main model state_dict.
# Call save_upi_masks() from rank 0 only in distributed (FSDP2) settings.
# ---------------------------------------------------------------------------

def collect_upi_masks(model):
    """
    Walk all modules and collect upi state keyed by layer_idx.

    Returns a dict:
      {
        layer_idx: {
          'type': 'trainable',
          'raw': Tensor,               # upi_scale_raw on CPU
          'target_multiplier': float,
        }
        or
        {
          'type': 'fixed',
          'mask': Tensor,              # upi_mask on CPU
        }
      }
    """
    result = {}
    for module in model.modules():
        if not hasattr(module, 'layer_idx') or module.layer_idx is None:
            continue
        idx = module.layer_idx
        upi_scale_raw = getattr(module, 'upi_scale_raw', None)
        upi_mask = getattr(module, 'upi_mask', None)
        if upi_scale_raw is not None:
            result[idx] = {
                'type': 'trainable',
                'raw': upi_scale_raw.detach().cpu(),
                'target_multiplier': module.upi_target_multiplier,
            }
        elif upi_mask is not None:
            result[idx] = {
                'type': 'fixed',
                'mask': upi_mask.detach().cpu(),
            }
    return result


def save_upi_masks(model, path):
    """
    Save upi masks / trainable scales for all layers to *path*.

    In multi-rank (FSDP2) runs call this from rank 0 only, after
    parameters have been gathered (e.g. inside save_single_file).
    """
    masks = collect_upi_masks(model)
    if masks:
        torch.save(masks, path)


def load_upi_masks(model, path_or_dict):
    """
    Restore upi state into matching model layers.

    *path_or_dict* can be a file path (str) or an already-loaded dict
    as returned by collect_upi_masks().
    """
    if isinstance(path_or_dict, str):
        masks = torch.load(path_or_dict, weights_only=False)
    else:
        masks = path_or_dict

    for module in model.modules():
        if not hasattr(module, 'layer_idx') or module.layer_idx is None:
            continue
        idx = module.layer_idx
        if idx not in masks:
            continue
        entry = masks[idx]
        upi_scale_raw = getattr(module, 'upi_scale_raw', None)
        upi_mask = getattr(module, 'upi_mask', None)
        if entry['type'] == 'trainable' and upi_scale_raw is not None:
            with torch.no_grad():
                upi_scale_raw.copy_(
                    entry['raw'].to(device=upi_scale_raw.device, dtype=upi_scale_raw.dtype)
                )
        elif entry['type'] == 'fixed' and upi_mask is not None:
            upi_mask.copy_(
                entry['mask'].to(device=upi_mask.device, dtype=upi_mask.dtype)
            )

# dt = scale_dt(self.upi_mask, dt, self.dt_bias)
def scale_dt(scale_mask, dt, dt_bias):
    assert (scale_mask.dim() == 0) or (
        scale_mask.dim() == 1 and scale_mask.size(0) == dt_bias.size(0)
    ), f"scale_mask must be scalar or of shape {(dt_bias.size(0),)}, got {tuple(scale_mask.shape)}"
    t = F.softplus(dt + dt_bias) / scale_mask
    y = torch.expm1(t).clamp_min(1e-6)
    return torch.log(y) - dt_bias

def dynamic_scale_mask(scale_mask, seq_max_new, seq_max=32768, seq_min=4096):
    scale = max(0, (seq_max_new-seq_min)/(seq_max-seq_min))
    return scale * (scale_mask - 1) + 1

# Easiest way to load/modify the mask: just change the state dict
def add_upi_to_state_dict(ckpt_dir, upi_dir, save_dir):
    state_dict = torch.load(ckpt_dir)
    upi_mask_dict = torch.load(upi_dir)
    for i in range(32):
        if i not in (9, 18, 27):
            state_dict['model_state'][f'backbone.layers.{i}.mixer.upi_mask'] = upi_mask_dict[i].to(torch.bfloat16)
    torch.save(state_dict, save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-dir', type=str, required=True)
    parser.add_argument('--upi-dir', type=str, required=True)
    parser.add_argument('--save-dir', type=str, required=True)
    args = parser.parse_args()

    add_upi_to_state_dict(
        ckpt_dir=args.ckpt_dir, 
        upi_dir=args.upi_dir, 
        save_dir=args.save_dir,
    )

    # add_upi_to_state_dict(
    #     ckpt_dir="/gpfs/davis/granites/bamba-merged/consolidated_ckpt.pth", 
    #     upi_dir="/gpfs/hshen/UPI_configs/upi_mask_layer.pt", 
    #     save_dir="/gpfs/hshen/bamba_upi_tune/bambav2_layer.pth",
    # )