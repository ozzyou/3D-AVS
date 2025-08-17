import pdb
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

import MinkowskiEngine as ME


class GlobalAttentionPool3d(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        output_dim: int = None,
    ):
        super().__init__()
        self.cpe = ME.MinkowskiConvolution(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=3,
            bias=True,
            dimension=3,
        )
        self.linear = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

        self.global_avg_pool = ME.MinkowskiGlobalAvgPooling()

        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, input_dict):
        return_attention = input_dict.get("return_attention", False)
        x = input_dict["minkunet_output"]

        # xCPE
        x_res = self.cpe(x).F
        x_res = self.linear(x_res)
        x_res = self.norm(x_res)
        x_res = ME.SparseTensor(
            x_res,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager,
        )
        x = x + x_res

        avg_feat = self.global_avg_pool(x).F

        bs = avg_feat.shape[0]
        clipcap_outputs = []
        for i in range(bs):
            query = avg_feat[i].unsqueeze(0).unsqueeze(0)  # (1, 1, C)
            key = x.decomposed_features[i].unsqueeze(1)  # (N, 1, C)

            output, weights = F.multi_head_attention_forward(
                query=query,
                key=key,
                value=key,
                embed_dim_to_check=x.shape[-1],
                num_heads=self.num_heads,
                q_proj_weight=self.q_proj.weight,
                k_proj_weight=self.k_proj.weight,
                v_proj_weight=self.v_proj.weight,
                in_proj_weight=None,
                in_proj_bias=torch.cat(
                    [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]
                ),
                bias_k=None,
                bias_v=None,
                add_zero_attn=False,
                dropout_p=0,
                out_proj_weight=self.c_proj.weight,
                out_proj_bias=self.c_proj.bias,
                use_separate_proj_weight=True,
                training=self.training,
                need_weights=return_attention,
            )
            clipcap_outputs.append(output.squeeze(1))

        input_dict["clipcap_output"] = torch.cat(clipcap_outputs, dim=0)

        return input_dict


class PerViewAttentionPool3d(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        num_heads: int = 8,
        output_dim: int = None,
        dataset_name: str = None,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        if input_dim != embed_dim:
            self.in_proj = ME.MinkowskiConvolution(
                in_channels=input_dim,
                out_channels=embed_dim,
                kernel_size=1,
                bias=True,
                dimension=3,
            )
        else:
            self.in_proj = None

        self.cpe = ME.MinkowskiConvolution(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=3,
            bias=True,
            dimension=3,
        )
        self.linear = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

        self.gap = nn.AdaptiveAvgPool2d((1, embed_dim))  # Global Average Pooling

        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.c_proj = nn.Linear(embed_dim, output_dim)
        self.num_heads = num_heads

    def forward(self, input_dict):
        return_attention = input_dict.get("return_attention", False)
        x = input_dict["minkunet_output"]
        mask = input_dict["mask"]  # mask = mask_chunk[vox_ind]
        camera_visible_mask = input_dict["camera_id_mask"]

        if self.in_proj is not None:
            x = self.in_proj(x)

        # xCPE
        x_res = self.cpe(x).F
        x_res = self.linear(x_res)
        x_res = self.norm(x_res)
        x_res = ME.SparseTensor(
            x_res,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager,
        )
        x = x + x_res

        bs = len(x.decomposed_features)
        n_views = camera_visible_mask.shape[1]
        div_10frame = 0
        div_curr = 0
        clipcap_outputs = []
        for i in range(bs):
            num_points_10frame = x.decomposed_coordinates[i].shape[0]
            div1 = div_10frame + num_points_10frame

            if self.dataset_name == "nuscenes":
                feat_curr = x.decomposed_features[i][mask[div_10frame:div1]]  # (N, C)
            else:
                feat_curr = x.decomposed_features[i]
            num_points_curr = feat_curr.shape[0]
            div2 = div_curr + num_points_curr

            sum_n = camera_visible_mask[div_curr:div2].sum(dim=0)
            max_n = torch.max(sum_n)

            avg_feat = torch.zeros((1, n_views, feat_curr.shape[1])).to(
                feat_curr.device
            )  # (1, n_views, C)
            input_feat = torch.zeros((n_views, max_n, feat_curr.shape[1])).to(
                feat_curr.device
            )  # (n_views, max_n, C)
            key_padding_mask = torch.zeros((n_views, max_n)).to(feat_curr.device)
            for j in range(n_views):
                feat_curr_j = feat_curr[
                    camera_visible_mask[div_curr:div2, j]
                ].unsqueeze(
                    0
                )  # (1, N_j, C)
                avg_feat[0, j, :] = self.gap(feat_curr_j)  # (1, 1, C)
                input_feat[j, : feat_curr_j.shape[1], :] = feat_curr_j
                key_padding_mask[j, feat_curr_j.shape[1] :] = 1

            query = avg_feat  # (1, 6, C)
            key = input_feat.transpose(1, 0)  # (N_j, 6, C)

            output, weights = F.multi_head_attention_forward(
                query=query,
                key=key,
                value=key,
                embed_dim_to_check=x.shape[-1],
                num_heads=self.num_heads,
                q_proj_weight=self.q_proj.weight,
                k_proj_weight=self.k_proj.weight,
                v_proj_weight=self.v_proj.weight,
                in_proj_weight=None,
                in_proj_bias=torch.cat(
                    [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]
                ),
                bias_k=None,
                bias_v=None,
                add_zero_attn=False,
                dropout_p=0,
                out_proj_weight=self.c_proj.weight,
                out_proj_bias=self.c_proj.bias,
                key_padding_mask=key_padding_mask,
                use_separate_proj_weight=True,
                training=self.training,
                need_weights=return_attention,
            )
            clipcap_outputs.append(output)

            div_10frame = div1
            div_curr = div2

        input_dict["output_smap"] = torch.cat(clipcap_outputs, dim=0)  # (B, 6, C)

        return input_dict
