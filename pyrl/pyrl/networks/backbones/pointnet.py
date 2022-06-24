import numpy as np
from copy import deepcopy
import torch, torch.nn as nn, torch.nn.functional as F

from .mlp import ConvMLP, LinearMLP
from ..builder import BACKBONES
from pyrl.utils.torch import ExtendedModule

from pytorch3d.transforms import quaternion_to_matrix


@BACKBONES.register_module()
class PointNet(ExtendedModule):
    def __init__(
        self,
        feat_dim,
        mlp_spec=[64, 128, 1024],
        global_feat=True,
        norm_cfg=dict(type="LN1d", eps=1e-6),
        act_cfg=dict(type="ReLU"),
    ):
        super(PointNet, self).__init__()
        self.global_feat = global_feat

        mlp_spec = deepcopy(mlp_spec)
        self.conv = ConvMLP(
            [
                feat_dim,
            ]
            + mlp_spec,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            inactivated_output=False,
        )

    def forward(self, inputs, object_feature=True, concat_state=None, **kwargs):
        xyz = inputs["xyz"] if isinstance(inputs, dict) else inputs
        assert not ("hand_pose" in kwargs.keys() and "obj_pose_info" in kwargs.keys())
        if "hand_pose" in kwargs.keys():
            hand_pose = kwargs.pop("hand_pose")
            # use the first hand to transform point cloud coordinates
            hand_xyz = hand_pose[:, 0, :3]
            hand_quat = hand_pose[:, 0, 3:]
            hand_rot = quaternion_to_matrix(hand_quat)
            xyz = xyz - hand_xyz[:, None, :]
            xyz = torch.einsum('bni,bij->bnj', xyz, hand_rot)   
        if "obj_pose_info" in kwargs.keys():
            obj_pose_info = kwargs.pop("obj_pose_info")
            center = obj_pose_info['center'] # [B, 3]
            xyz = xyz - center[:, None, :]
            if 'rot' in obj_pose_info.keys():
                rot = obj_pose_info['rot'] # [B, 3, 3]
                xyz = torch.einsum('bni,bij->bnj', xyz, rot)  

        with torch.no_grad():
            if isinstance(inputs, dict):
                feature = [xyz]
                if "rgb" in inputs:
                    feature.append(inputs["rgb"])
                if "seg" in inputs:
                    feature.append(inputs["seg"])
                if concat_state is not None: # [B, C]
                    feature.append(concat_state[:, None, :].expand(-1, xyz.shape[1], -1))
                feature = torch.cat(feature, dim=-1)
            else:
                feature = xyz

            feature = feature.permute(0, 2, 1).contiguous()
        feature = self.conv(feature)
        if self.global_feat:
            feature = feature.max(-1)[0]
        else:
            gl_feature = feature.max(-1, keepdims=True)[0].repeat(1, 1, feature.shape[-1])
            feature = torch.cat([feature, gl_feature], dim=1)
        return feature

