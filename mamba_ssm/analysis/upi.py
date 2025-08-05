import argparse

import torch
import torch.nn.functional as F

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