# Copyright (c) 2024, Tri Dao, Albert Gu.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from causal_conv1d.causal_conv1d_varlen import causal_conv1d_varlen_states
except ImportError:
    causal_conv1d_varlen_states = None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated

from mamba_ssm.distributed.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from mamba_ssm.distributed.distributed_utils import all_reduce, reduce_scatter

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined

from huggingface_hub import PyTorchModelHubMixin

import os 
from mamba_ssm.analysis.LongMamba import get_top_k_token_indices, get_topk_mask_channelwise, get_channelwise_topAlpha, get_channelwise_topBound, get_channelwise_offline, get_channelwise_normalize, get_channelwise_dt_threshold, merge_config
from mamba_ssm.analysis.mmd import mmd_from_gqa_inputs, mmd_from_ssd_inputs

class Mamba2(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        d_model,
        d_state=128,
        d_conv=4,
        conv_init=None,
        expand=2,
        headdim=64,
        d_ssm=None,  # If not None, we only apply SSM on this many dimensions, the rest uses gated MLP
        ngroups=1,
        A_init_range=(1, 16),
        D_has_hdim=False,
        rmsnorm=True,
        norm_before_gate=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        bias=False,
        conv_bias=True,
        # Fused kernel and sharding options
        chunk_size=256,
        use_mem_eff_path=True,
        layer_idx=None,  # Absorb kwarg for general module
        process_group=None,
        sequence_parallel=True,
        exp_type={},
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.process_group = process_group
        self.sequence_parallel = sequence_parallel
        self.world_size = 1 if process_group is None else process_group.size()
        self.local_rank = 0 if process_group is None else process_group.rank()
        self.d_inner = (self.expand * self.d_model) // self.world_size
        assert self.d_inner * self.world_size == self.expand * self.d_model
        self.headdim = headdim
        self.d_ssm = self.d_inner if d_ssm is None else d_ssm // self.world_size
        assert ngroups % self.world_size == 0
        self.ngroups = ngroups // self.world_size
        assert self.d_ssm % self.headdim == 0
        self.nheads = self.d_ssm // self.headdim
        self.D_has_hdim = D_has_hdim
        self.rmsnorm = rmsnorm
        self.norm_before_gate = norm_before_gate
        self.dt_limit = dt_limit
        self.activation = "silu"
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx

        self.exp_type = exp_type

        if "context_length" not in self.exp_type.keys():
            self.exp_type["context_length"] = 4096

        # load head mask for scalingAdd commentMore actions
        if "head" in self.exp_type.keys():
            data = {}
            with open(self.exp_type["head"], "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)  
                    data.update(obj)
            self.head_mask = torch.tensor(data[f"{self.layer_idx}"], dtype=torch.bool)

        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        if self.process_group is None:
            self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)
        else:
            self.in_proj = ColumnParallelLinear(self.d_model, d_in_proj * self.world_size, bias=bias,
                                                process_group=self.process_group, sequence_parallel=self.sequence_parallel,
                                                **factory_kwargs)

        conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)

        self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_ssm if self.D_has_hdim else self.nheads, device=device))
        self.D._no_weight_decay = True

        if self.rmsnorm:
            assert RMSNormGated is not None
            self.norm = RMSNormGated(self.d_ssm, eps=1e-5, norm_before_gate=self.norm_before_gate,
                                     group_size=self.d_ssm // ngroups, **factory_kwargs)

        if self.process_group is None:
            self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        else:
            self.out_proj = RowParallelLinear(self.d_inner * self.world_size, self.d_model, bias=bias,
                                              process_group=self.process_group, sequence_parallel=self.sequence_parallel,
                                              **factory_kwargs)

    def forward(self, u, seqlen=None, seq_idx=None, cu_seqlens=None, inference_params=None):
        """
        u: (batch, seqlen, hidden_dim) if seqlen=None.
            If seqlen is not None, u is (batch * seqlen, hidden_dim). This is so that when we
            split u during sequence parallel, we split the batch * seqlen dimension
            (in case batch is small).
        Returns: same shape as u
        """
        self.seq_len = seqlen
        seqlen_og = seqlen
        if seqlen is None:
            batch, seqlen, dim = u.shape
        else:
            batch_seqlen, dim = u.shape
            batch = batch_seqlen // seqlen

        conv_state, ssm_state = None, None
        if inference_params is not None:
            inference_batch = cu_seqlens.shape[0] - 1 if cu_seqlens is not None else batch
            conv_state, ssm_state = self._get_states_from_cache(inference_params, inference_batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(u, conv_state, ssm_state)
                return out

        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj) or (B * L, d_in_proj)
        if seqlen_og is not None:
            zxbcdt = rearrange(zxbcdt, "(b l) d -> b l d", l=seqlen)
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        A = -torch.exp(self.A_log.float())  # (nheads) or (d_inner, d_state)
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)
        if self.use_mem_eff_path and inference_params is None:
            assert False, "Warning: scaling not implemented!"
            out = mamba_split_conv1d_scan_combined(
                zxbcdt,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.dt_bias,
                A,
                D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
                chunk_size=self.chunk_size,
                seq_idx=seq_idx,
                activation=self.activation,
                rmsnorm_weight=self.norm.weight if self.rmsnorm else None,
                rmsnorm_eps=self.norm.eps if self.rmsnorm else 1e-6,
                outproj_weight=self.out_proj.weight,
                outproj_bias=self.out_proj.bias,
                headdim=None if self.D_has_hdim else self.headdim,
                ngroups=self.ngroups,
                norm_before_gate=self.norm_before_gate,
                **dt_limit_kwargs,
            )
            if seqlen_og is not None:
                out = rearrange(out, "b l d -> (b l) d")
            if self.process_group is not None:
                reduce_fn = reduce_scatter if self.sequence_parallel else all_reduce
                out = reduce_fn(out, self.process_group)
        else:
            d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
            z0, x0, z, xBC, dt = torch.split(
                zxbcdt,
                [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
                dim=-1
            )
            if conv_state is not None:
                if cu_seqlens is None:
                    # If we just take xBC[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                    # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                    xBC_t = rearrange(xBC, "b l d -> b d l")
                    conv_state.copy_(F.pad(xBC_t, (self.d_conv - xBC_t.shape[-1], 0)))  # Update state (B D W)
                else:
                    assert causal_conv1d_varlen_states is not None, "varlen inference requires causal_conv1d package"
                    assert batch == 1, "varlen inference only supports batch dimension 1"
                    conv_varlen_states = causal_conv1d_varlen_states(
                        xBC.squeeze(0), cu_seqlens, state_len=conv_state.shape[-1]
                    )
                    conv_state.copy_(conv_varlen_states)
            assert self.activation in ["silu", "swish"]
            if causal_conv1d_fn is None or self.activation not in ["silu", "swish"]:
                assert seq_idx is None, "varlen conv1d requires the causal_conv1d package"
                xBC = self.act(
                    self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, :-(self.d_conv - 1)]
                )  # (B, L, self.d_ssm + 2 * ngroups * d_state)
            else:
                xBC = causal_conv1d_fn(
                    xBC.transpose(1, 2),
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    seq_idx=seq_idx,
                ).transpose(1, 2)
            x, B, C = torch.split(xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
            
            if "layer" in self.exp_type.keys():
                if self.layer_idx in self.exp_type["layer"]:
                    A = A.clone()
                    A *= int(self.exp_type["context_length"])/self.seq_len
                    B = B.clone()
                    B *= int(self.exp_type["context_length"])/self.seq_len

            if "head" in self.exp_type.keys():
                A = A.clone()
                A[..., self.head_mask.to(A.device)] *= int(self.exp_type["context_length"])/self.seq_len
                B = B.clone()
                B[..., self.head_mask.to(B.device)] *= int(self.exp_type["context_length"])/self.seq_len

            if "erf" in self.exp_type.keys():
                mmd = mmd_from_ssd_inputs(
                    dt, 
                    A, 
                    B.view(batch_size, seq_len, self.n_groups, -1), 
                    C.view(batch_size, seq_len, self.n_groups, -1), 
                    dt_bias=self.dt_bias, 
                    dt_softplus=True, 
                    dt_limit=(0.0, float("inf"))
                    )
                record = {
                    "layer_idx": self.layer_idx,
                    "seq_len": self.seq_len,
                    "erf": mmd.tolist()
                    }
                with open(self.exp_type["erf"], 'a') as f:
                    f.write(json.dumps(record) + '\n')
                
            if "logits" in self.exp_type.keys():
                curr_state = {}
                curr_state['hidden_states'] = hidden_states.view(batch_size, seq_len, -1, self.head_dim)
                curr_state['dt'] = dt
                curr_state['A'] = A
                curr_state['B'] = B.view(batch_size, seq_len, self.n_groups, -1)
                curr_state['C'] = C.view(batch_size, seq_len, self.n_groups, -1)
                curr_state['D'] = self.D
                curr_state['dt_bias'] = self.dt_bias
                torch.save(curr_state, os.path.join(self.exp_type["logits"], f'curr_state_{seq_len}_{self.layer_idx}.pt'))

            dt_bias = self.dt_bias
            dt_softplus = True

            if "longmamba" in self.exp_type.keys():
                # dt alignment
                params_for_debug = {}
                dt = nn.functional.softplus(dt + self.dt_bias)
                if False:
                # if merge_config is not None and merge_config['model_arch'] == "ours" and int(self.seq_len/1e3) > 2:
                    channel_threshold = merge_config['c']
                    tA_prod_path = f"./artifacts/{merge_config['align_path']}/tA_prod/tA_prod_layer_{self.layer_idx}.pt"
                    alpha_path = f"./artifacts/{merge_config['align_path']}/alpha/alpha_layer_{self.layer_idx}.pt"
                    decay_path = f"./artifacts/{merge_config['align_path']}/decay/decay_layer_{self.layer_idx}.pt"
                    dt_thre_path = f"./artifacts/{merge_config['align_path']}/delta_t-thre/delta_t-thre_layer_{self.layer_idx}.pt"
                    tA_prod = torch.load(tA_prod_path, map_location=dt.device)
                    self.channel_mask = tA_prod > channel_threshold
                    whether_mask = self.channel_mask.sum() != 0
                    available_values = []
                    if merge_config['our_method'] in ["alpha", "offline"]:
                        alpha_all = torch.load(alpha_path, map_location=dt.device)
                        for k in alpha_all:
                            available_values.append(int(k[:-1])*1e3)
                    elif merge_config['our_method'] in ["bound", "norm"]:  
                        decay = torch.load(decay_path, map_location=dt.device)
                    elif merge_config['our_method'] in ["dt_thre"]:
                        dt_thre_all = torch.load(dt_thre_path, map_location=dt.device)
                        for k in dt_thre_all:
                            available_values.append(int(k[:-1])*1e3)
                        
                    self.slow_factor = 0
                    self.topk = 2000
                    dt = rearrange(dt, "b l h -> b h l")
                    if merge_config['our_method'] == "channelwise_topk" and whether_mask:
                        topk_mask = get_topk_mask_channelwise(delta_t=dt[:, self.channel_mask], k=self.topk)
                        dt[:, self.channel_mask] = torch.where(topk_mask, dt[:, self.channel_mask], dt[:, self.channel_mask] * self.slow_factor)
                    elif merge_config['our_method'] == "all_topk" and whether_mask:
                        topk_indice = get_top_k_token_indices(dt[:, self.channel_mask], k=self.topk)
                        topk_mask = torch.zeros_like(dt[:, self.channel_mask], dtype=torch.bool)
                        topk_mask[:,:,topk_indice] = True
                        dt[:, self.channel_mask] = torch.where(topk_mask, dt[:, self.channel_mask], dt[:, self.channel_mask] * self.slow_factor)
                    elif whether_mask and merge_config['b'] != 0:
                        key_num = int(min(available_values, key=lambda x: abs(self.seqlen - x))/1e3) if available_values != [] else None
                        if "alpha" in merge_config['our_method']:
                            channel_alpha = alpha_all[f"{key_num}k"].to(dt.device)
                            topk_mask = get_channelwise_topAlpha(delta_t=dt[:, self.channel_mask], alpha=channel_alpha[self.channel_mask]*merge_config['b'])
                            dt[:, self.channel_mask] = torch.where(topk_mask, dt[:, self.channel_mask], dt[:, self.channel_mask] * self.slow_factor)
                        elif "offline" in merge_config['our_method']:
                            key_num_offline = int(min(available_values, key=lambda x: abs(self.seqlen - x))/1e3)
                            channel_alpha = alpha_all[f"{key_num_offline}k"].to(dt.device)
                            topk_mask, _ = get_channelwise_offline(delta_t=dt[:, self.channel_mask], alpha=channel_alpha[self.channel_mask]*merge_config['b'])
                            dt[:, self.channel_mask] = torch.where(topk_mask, dt[:, self.channel_mask], dt[:, self.channel_mask] * self.slow_factor)
                        elif "bound" in merge_config['our_method']:
                            topk_mask = get_channelwise_topBound(delta_t=dt[:, self.channel_mask], decay=decay[self.channel_mask]*merge_config['b'])
                            dt[:, self.channel_mask] = torch.where(topk_mask, dt[:, self.channel_mask], dt[:, self.channel_mask] * self.slow_factor)
                        elif "norm" in merge_config['our_method']:
                            dt_norm = get_channelwise_normalize(delta_t=dt[:, self.channel_mask], decay=decay[self.channel_mask]*merge_config['b'])
                            dt[:, self.channel_mask] = dt_norm
                        elif "dt_thre" in merge_config['our_method']:
                            channel_dt_thre_all = dt_thre_all[f"{key_num}k"].to(dt.device)
                            # self.dt_thre[f"layer_{self.layer_idx}"] = channel_dt_thre_all
                            topk_mask = get_channelwise_dt_threshold(delta_t=dt[:, self.channel_mask], dt_thre=channel_dt_thre_all[self.channel_mask]*merge_config['b'])
                            dt[:, self.channel_mask] = torch.where(topk_mask, dt[:, self.channel_mask], dt[:, self.channel_mask] * self.slow_factor)
                    else:
                        if whether_mask:
                            print("Warning: no method is applied")
                        dt[:, self.channel_mask] = dt[:, self.channel_mask] * 0
                    dt = rearrange(dt, "b h l -> b l h")
                if merge_config['save_para4debug']:
                    params_for_debug['A'] = A.clone().cpu()
                    params_for_debug['Sb_x'] = B.clone().cpu()  # B before discretization
                    params_for_debug['C'] = C.clone().cpu()
                    params_for_debug['delta_t'] = dt.clone().cpu()
                    print("Save parameters mamba2 debug")
                    params_for_debug['B_t'] = None
                
                dt_bias = None
                dt_softplus = False
                # Change params_for_debug from outputing to saving to log file
                with open(self.exp_type["longmamba"], 'a') as f:
                    f.write(json.dumps(params_for_debug) + '\n')

            y = mamba_chunk_scan_combined(
                rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
                dt,
                A,
                rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
                rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
                chunk_size=self.chunk_size,
                D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
                z=rearrange(z, "b l (h p) -> b l h p", p=self.headdim) if not self.rmsnorm else None,
                dt_bias=dt_bias,
                dt_softplus=dt_softplus,
                seq_idx=seq_idx,
                cu_seqlens=cu_seqlens,
                **dt_limit_kwargs,
                return_final_states=ssm_state is not None,
                return_varlen_states=cu_seqlens is not None and inference_params is not None,
            )
            if ssm_state is not None:
                y, last_state, *rest = y
                if cu_seqlens is None:
                    ssm_state.copy_(last_state)
                else:
                    varlen_states = rest[0]
                    ssm_state.copy_(varlen_states)
            y = rearrange(y, "b l h p -> b l (h p)")
            if self.rmsnorm:
                y = self.norm(y, z)
            if d_mlp > 0:
                y = torch.cat([F.silu(z0) * x0, y], dim=-1)
            if seqlen_og is not None:
                y = rearrange(y, "b l d -> (b l) d")
            out = self.out_proj(y)
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        zxbcdt = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1
        )

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = xBC
            xBC = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                xBC = xBC + self.conv1d.bias
            xBC = self.act(xBC).to(dtype=dtype)
        else:
            xBC = causal_conv1d_update(
                xBC,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x, B, C = torch.split(xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        A = -torch.exp(self.A_log.float())  # (nheads,)

        if "layer" in self.exp_type.keys():
            if self.layer_idx in self.exp_type["layer"]:
                A = A.clone()
                A *= int(self.exp_type["context_length"])/self.seq_len
                B = B.clone()
                B *= int(self.exp_type["context_length"])/self.seq_len

        if "head" in self.exp_type.keys():
            A = A.clone()
            A[..., self.head_mask.to(A.device)] *= int(self.exp_type["context_length"])/self.seq_len
            B = B.clone()
            B[..., self.head_mask.to(B.device)] *= int(self.exp_type["context_length"])/self.seq_len

        dt_bias = self.dt_bias
        dt_softplus = True

        if "longmamba" in self.exp_type.keys():
            # dt alignment
            params_for_debug = {}
            dt = nn.functional.softplus(dt + self.dt_bias)
            if merge_config is not None and merge_config['model_arch'] == "ours" and int(self.seq_len/1e3) > 2:
                channel_threshold = merge_config['c']
                tA_prod_path = f"./artifacts/{merge_config['align_path']}/tA_prod/tA_prod_layer_{self.layer_idx}.pt"
                alpha_path = f"./artifacts/{merge_config['align_path']}/alpha/alpha_layer_{self.layer_idx}.pt"
                decay_path = f"./artifacts/{merge_config['align_path']}/decay/decay_layer_{self.layer_idx}.pt"
                dt_thre_path = f"./artifacts/{merge_config['align_path']}/delta_t-thre/delta_t-thre_layer_{self.layer_idx}.pt"
                tA_prod = torch.load(tA_prod_path, map_location=dt.device)
                self.channel_mask = tA_prod > channel_threshold
                whether_mask = self.channel_mask.sum() != 0
                available_values = []
                if merge_config['our_method'] in ["alpha", "offline"]:
                    alpha_all = torch.load(alpha_path, map_location=dt.device)
                    for k in alpha_all:
                        available_values.append(int(k[:-1])*1e3)
                elif merge_config['our_method'] in ["bound", "norm"]:  
                    decay = torch.load(decay_path, map_location=dt.device)
                elif merge_config['our_method'] in ["dt_thre"]:
                    dt_thre_all = torch.load(dt_thre_path, map_location=dt.device)
                    for k in dt_thre_all:
                        available_values.append(int(k[:-1])*1e3)
                    
                self.slow_factor = 0
                self.topk = 2000
                dt = rearrange(dt, "b l h -> b h l")
                if merge_config['our_method'] == "channelwise_topk" and whether_mask:
                    topk_mask = get_topk_mask_channelwise(delta_t=dt[:, self.channel_mask], k=self.topk)
                    dt[:, self.channel_mask] = torch.where(topk_mask, dt[:, self.channel_mask], dt[:, self.channel_mask] * self.slow_factor)
                elif merge_config['our_method'] == "all_topk" and whether_mask:
                    topk_indice = get_top_k_token_indices(dt[:, self.channel_mask], k=self.topk)
                    topk_mask = torch.zeros_like(dt[:, self.channel_mask], dtype=torch.bool)
                    topk_mask[:,:,topk_indice] = True
                    dt[:, self.channel_mask] = torch.where(topk_mask, dt[:, self.channel_mask], dt[:, self.channel_mask] * self.slow_factor)
                elif whether_mask and merge_config['b'] != 0:
                    key_num = int(min(available_values, key=lambda x: abs(self.seqlen - x))/1e3) if available_values != [] else None
                    if "alpha" in merge_config['our_method']:
                        channel_alpha = alpha_all[f"{key_num}k"].to(dt.device)
                        topk_mask = get_channelwise_topAlpha(delta_t=dt[:, self.channel_mask], alpha=channel_alpha[self.channel_mask]*merge_config['b'])
                        dt[:, self.channel_mask] = torch.where(topk_mask, dt[:, self.channel_mask], dt[:, self.channel_mask] * self.slow_factor)
                    elif "offline" in merge_config['our_method']:
                        key_num_offline = int(min(available_values, key=lambda x: abs(self.seqlen - x))/1e3)
                        channel_alpha = alpha_all[f"{key_num_offline}k"].to(dt.device)
                        topk_mask, _ = get_channelwise_offline(delta_t=dt[:, self.channel_mask], alpha=channel_alpha[self.channel_mask]*merge_config['b'])
                        dt[:, self.channel_mask] = torch.where(topk_mask, dt[:, self.channel_mask], dt[:, self.channel_mask] * self.slow_factor)
                    elif "bound" in merge_config['our_method']:
                        topk_mask = get_channelwise_topBound(delta_t=dt[:, self.channel_mask], decay=decay[self.channel_mask]*merge_config['b'])
                        dt[:, self.channel_mask] = torch.where(topk_mask, dt[:, self.channel_mask], dt[:, self.channel_mask] * self.slow_factor)
                    elif "norm" in merge_config['our_method']:
                        dt_norm = get_channelwise_normalize(delta_t=dt[:, self.channel_mask], decay=decay[self.channel_mask]*merge_config['b'])
                        dt[:, self.channel_mask] = dt_norm
                    elif "dt_thre" in merge_config['our_method']:
                        channel_dt_thre_all = dt_thre_all[f"{key_num}k"].to(dt.device)
                        # self.dt_thre[f"layer_{self.layer_idx}"] = channel_dt_thre_all
                        topk_mask = get_channelwise_dt_threshold(delta_t=dt[:, self.channel_mask], dt_thre=channel_dt_thre_all[self.channel_mask]*merge_config['b'])
                        dt[:, self.channel_mask] = torch.where(topk_mask, dt[:, self.channel_mask], dt[:, self.channel_mask] * self.slow_factor)
                else:
                    if whether_mask:
                        print("Warning: no method is applied")
                    dt[:, self.channel_mask] = dt[:, self.channel_mask] * 0
                dt = rearrange(dt, "b h l -> b l h")
            if merge_config['save_para4debug']:
                params_for_debug['A'] = A.clone().cpu()
                params_for_debug['Sb_x'] = B.clone().cpu()  # B before discretization
                params_for_debug['C'] = C.clone().cpu()
                params_for_debug['delta_t'] = dt.clone().cpu()
                print("Save parameters mamba2 debug")
                params_for_debug['B_t'] = None
            
            dt_bias = None
            dt_softplus = False
            # Change params_for_debug from outputing to saving to log file
            with open(self.exp_type["longmamba"], 'a') as f:
                f.write(json.dumps(params_for_debug) + '\n')

        # SSM step
        if selective_state_update is None:
            assert self.ngroups == 1, "Only support ngroups=1 for this inference code path"
            # Discretize A and B
            dt = F.softplus(dt + self.dt_bias.to(dtype=dt.dtype))  # (batch, nheads)
            dA = torch.exp(dt * A)  # (batch, nheads)
            x = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            dBx = torch.einsum("bh,bn,bhp->bhpn", dt, B, x)
            ssm_state.copy_(ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx)
            y = torch.einsum("bhpn,bn->bhp", ssm_state.to(dtype), C)
            y = y + rearrange(self.D.to(dtype), "h -> h 1") * x
            y = rearrange(y, "b h p -> b (h p)")
            if not self.rmsnorm:
                y = y * self.act(z)  # (B D)
        else:
            A = repeat(A, "h -> h p n", p=self.headdim, n=self.d_state).to(dtype=torch.float32)
            dt = repeat(dt, "b h -> b h p", p=self.headdim)
            dt_bias = repeat(self.dt_bias, "h -> h p", p=self.headdim)
            D = repeat(self.D, "h -> h p", p=self.headdim)
            B = rearrange(B, "b (g n) -> b g n", g=self.ngroups)
            C = rearrange(C, "b (g n) -> b g n", g=self.ngroups)
            x_reshaped = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            if not self.rmsnorm:
                z = rearrange(z, "b (h p) -> b h p", p=self.headdim)
            y = selective_state_update(
                ssm_state, x_reshaped, dt, A, B, C, D, z=z if not self.rmsnorm else None,
                dt_bias=dt_bias, dt_softplus=True
            )
            y = rearrange(y, "b h p -> b (h p)")
        if self.rmsnorm:
            y = self.norm(y, z)
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)
        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_conv, self.conv1d.weight.shape[0], device=device, dtype=conv_dtype
        ).transpose(1, 2)
        ssm_dtype = self.in_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size, self.nheads, self.headdim, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_conv,
                self.conv1d.weight.shape[0],
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            ).transpose(1, 2)
            ssm_state = torch.zeros(
                batch_size,
                self.nheads,
                self.headdim,
                self.d_state,
                device=self.in_proj.weight.device,
                dtype=self.in_proj.weight.dtype,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state
