import torch
import torch.nn.functional as F

# dt *= scale_dt(self.upi_mask, dt, self.dt_bias)
def scale_dt(scale_mask, dt, dt_bias):
    assert (scale_mask.dim() == 0) or (
        scale_mask.dim() == 1 and scale_mask.size(0) == dt_bias.size(0)
    ), f"scale_mask must be scalar or of shape {(dt_bias.size(0),)}, got {tuple(scale_mask.shape)}"
    t = scale_mask * F.softplus(dt + dt_bias)
    y = torch.expm1(t).clamp_min(1e-6)
    return (torch.log(y) - dt_bias) / dt

def add_upi_to_ckpt(ckpt_dir, upi_dir, save_dir):
    state_dict = torch.load(ckpt_dir)
    upi_mask_dict = torch.load(upi_dir)
    for i in range(32):
        if i not in (9, 18, 27):
            state_dict['model_state'][f'backbone.layers.{i}.mixer.upi_mask'] = upi_mask_dict[i].to(torch.bfloat16)
    print(state_dict)
    torch.save(state_dict, save_dir)

if __name__ == "__main__":
    add_upi_to_ckpt(
        ckpt_dir="/gpfs/davis/granites/bamba-merged/consolidated_ckpt.pth", 
        upi_dir="/gpfs/hshen/UPI_configs/upi_mask_10.pt", 
        save_dir="/gpfs/hshen/bamba_upi_tune/bambav2_base.pth",
    )