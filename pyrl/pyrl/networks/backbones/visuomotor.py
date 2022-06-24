"""
End-to-End Training of Deep Visuomotor Policies
    https://arxiv.org/pdf/1504.00702.pdf
Visuomotor as the base class of all visual polices.
"""
from pyrl.utils.meta import get_logger
import torch, torch.nn as nn
from pyrl.utils.torch import ExtendedModule, ExtendedModuleList, freeze_params, unfreeze_params
from pyrl.utils.data import GDict, DictArray, recover_with_mask, is_seq_of
from ..builder import build_model, BACKBONES
from copy import copy, deepcopy


def get_obj_pose_info_from_obs(obs, idx, use_rot):
    assert idx is not None and isinstance(idx, list)
    if "target_box" in obs: # for OpenCabinetDrawer and OpenCabinetDoor, since a cabinet has several links, so we need to get the target link
        obj_pose_info = obs['target_box']
    elif "inst_box" in obs:
        obj_pose_info = obs['inst_box']
    else:
        raise NotImplementedError("For ManiSkill1 envs, please ensure that env_cfg.with_3d_ann=True")
    obj_pose_info = {'center': obj_pose_info[0][:, idx, :], 'rot': obj_pose_info[2][:, idx, :, :]} # center [B, Ninst, 3]; rot [B, Ninst, 3, 3]
    if not use_rot:
        obj_pose_info.pop('rot')
    return obj_pose_info     


@BACKBONES.register_module()
class Visuomotor(ExtendedModule):
    def __init__(self, visual_nn_cfg, mlp_cfg, visual_nn=None, 
                 obj_frame=-1, obj_frame_rot=False,
                 rnn=None, freeze_visual_nn=False, freeze_mlp=False):
        """
        obj_frame: int
            If obj_frame >= 0, this refers to the semantic id of obj frame to be selected to transform the point clouds  
        obj_frame_rot: bool
            If obj_frame >= 0, then if obj_frame_rot, use object-centric rotation to transform the point clouds
        """             
        super(Visuomotor, self).__init__()
        # Feature extractor [Can be shared with other network]
        self.visual_nn = build_model(visual_nn_cfg) if visual_nn is None else visual_nn
        if freeze_visual_nn:
            get_logger().warning("We freeze the visual backbone!")
            freeze_params(self.visual_nn)

        self.final_mlp = build_model(mlp_cfg)
        if freeze_mlp:
            get_logger().warning("We freeze the whole mlp part!")
            from .mlp import LinearMLP

            assert isinstance(self.final_mlp, LinearMLP), "The final mlp should have type LinearMLP."
            freeze_params(self.final_mlp)

        self.saved_feature = None
        self.saved_visual_feature = None

        assert isinstance(obj_frame, int)
        self.obj_frame = obj_frame
        self.obj_frame_rot = obj_frame_rot

    def forward(
        self,
        obs,
        feature=None,
        visual_feature=None,
        save_feature=False,
        with_robot_state=True,
        rnn_states=None,
        episode_dones=None,
        is_valid=None,
        rnn_mode="base",
        **kwargs,
    ):
        assert isinstance(obs, dict), f"obs is not a dict! {type(obs)}"
        assert not (feature is not None and visual_feature is not None), f"You cannot provide visual_feature and feature at the same time!"
        self.saved_feature = None
        self.saved_visual_feature = None
        save_feature = save_feature or (feature is not None or visual_feature is not None)
        obs_keys = obs.keys()
        obs = copy(obs)

        hand_pose = obs.pop("hand_pose", None)
        robot_state = None
        states = None
        for key in ["state", "agent"]:
            if key in obs:
                assert robot_state is None, f"Please provide only one robot state! Obs Keys: {obs_keys}"
                robot_state = obs.pop(key)

        if self.obj_frame >= 0:
            obj_pose_info = get_obj_pose_info_from_obs(obs, [self.obj_frame], use_rot=self.obj_frame_rot)
            obj_pose_info['center'] = obj_pose_info['center'].squeeze(1)
            if 'rot' in obj_pose_info.keys():
                obj_pose_info['rot'] = obj_pose_info['rot'].squeeze(1)
        else:
            obj_pose_info = None


        for key in obs_keys:
            if "_box" in key or "_seg" in key or key in ["point_to_inst_sem_label", "point_to_target_sem_label"]:
                obs.pop(key)
        if not ("xyz" in obs or "rgb" in obs or "rgbd" in obs):
            assert len(obs) == 1, f"Observations need to contain only one visual element! Obs Keys: {obs.keys()}!"
            obs = obs[list(obs.keys())[0]]

        if feature is None:
            if visual_feature is None:
                if not self.is_recurrent:
                    assert not (hand_pose is not None and obj_pose_info is not None)
                    if hand_pose is None and obj_pose_info is None:
                        feat = self.visual_nn(obs)
                    elif hand_pose is not None:
                        feat = self.visual_nn(obs, hand_pose=hand_pose)
                    elif obj_pose_info is not None:
                        feat = self.visual_nn(obs, obj_pose_info=obj_pose_info)
                else:
                    assert hand_pose is None, "hand_pose is not supported in RNN policy yet"
                    assert obj_pose_info is None, "obj_pose_info is not supported in RNN policy yet"
                    obs_dict = DictArray(obs)

                    # Convert sequential observations to normal ones.
                    if is_valid is None:
                        if "xyz" in obs_dict:
                            # pointcloud
                            valid_shape = obs_dict["xyz"].shape[:-2]
                        else:
                            raise NotImplementedError

                        is_valid = torch.ones(*valid_shape, dtype=torch.bool, device=self.device)
                    else:
                        is_valid = is_valid > 0.5

                    compact_obs = obs_dict.select_with_mask(is_valid, wrapper=False)
                    feat = self.visual_nn(compact_obs)
                    feat = recover_with_mask(feat, is_valid)

                    # print(robot_state, GDict(obs).shape)
                    # exit(0)
                    if robot_state is not None:
                        # print(feat.shape, robot_state.shape)
                        # exit(0)
                        # robot_state =
                        # base vel and vase angle vel
                        feat = torch.cat([feat, robot_state[..., :3]], dim=-1)
            else:
                feat = visual_feature

            if self.rnn is not None:
                feat = self.rnn(feat, rnn_states=rnn_states, episode_dones=episode_dones, rnn_mode=rnn_mode)
                if rnn_mode != "base":
                    feat, states = feat

            if save_feature:
                self.saved_visual_feature = feat.clone()

            if robot_state is not None and with_robot_state:
                assert feat.ndim == robot_state.ndim, "Visual feature and state vector should have the same dimension!"
                feat = torch.cat([feat, robot_state], dim=-1)

            if save_feature:
                self.saved_feature = feat.clone()
        else:
            feat = feature

        if self.final_mlp is not None:
            feat = self.final_mlp(feat)

        return (feat, states) if (rnn_mode != "base" and self.is_recurrent) else feat




@BACKBONES.register_module()
class VisuomotorTransformerFrame(ExtendedModule):
    def __init__(self, visual_nn_cfg, mlp_cfg, is_value=False, visual_nn=None, freeze_visual_nn=False,
                 late_concat_state=True,
                 aux_regress=None, aux_regress_coeff=0.5, action_hierarchy=None, mix_action=False, 
                 fuse_feature_single_action=False, 
                 use_obj_frames=None, obj_frame_rot=False, 
                 debug=False):
        super(VisuomotorTransformerFrame, self).__init__()
        self.use_obj_frames = use_obj_frames
        if self.use_obj_frames is not None and visual_nn_cfg is not None:
            assert isinstance(self.use_obj_frames, list) and isinstance(self.use_obj_frames[0], int)
            visual_nn_cfg['num_obj_frames'] = len(self.use_obj_frames)       
        self.obj_frame_rot = obj_frame_rot             

        # Feature extractor [Can be shared with other network]
        self.visual_nn = build_model(visual_nn_cfg) if visual_nn is None else visual_nn
        self.Nframe = self.visual_nn.num_frames
        # if mix_action, we don't ignore object frames for action prediction, since in mix_action we predict the entire action space, 
        # not the action space belonging to a particular frame
        if self.use_obj_frames is not None and mix_action == False: 
            self.Nframe_m_obj = self.Nframe - len(self.use_obj_frames) # base and hand frames
        else:
            self.Nframe_m_obj = self.Nframe     
        if freeze_visual_nn:
            get_logger().warning("We freeze the visual backbone!")
            freeze_params(self.visual_nn)  

        self.late_concat_state = late_concat_state

        self.aux_regress = aux_regress
        assert self.aux_regress in [None, 'partial', 'full']
        self.aux_regress_coeff = aux_regress_coeff
        self.action_hierarchy = action_hierarchy
        assert self.action_hierarchy in [None, 'b2h', 'h2b']
        self.mix_action = mix_action
        self.fuse_feature_single_action = fuse_feature_single_action
        assert (int(self.aux_regress is not None) + int(self.action_hierarchy is not None) 
            + int(self.mix_action != False) + int(self.fuse_feature_single_action) <= 1), (
            "Only choose at most one option among aux_regress, action_hierarchy, mix_action, and fuse_feature_single_action!"
        )

        self.base_action_dim = 4
        self.per_hand_action_dim = 9
        if self.use_obj_frames is None or mix_action == False:
            self.full_action_dim = self.base_action_dim + (self.Nframe_m_obj - 1) * self.per_hand_action_dim # dim of the entire action space
        else: # if mix_action and use_obj_frames, then Nframe_m_obj == Nframe
            self.full_action_dim = self.base_action_dim + (self.Nframe_m_obj - len(self.use_obj_frames) - 1) * self.per_hand_action_dim
        if self.mix_action and not is_value:
            if self.mix_action == True:
                self.mix_action_params = nn.Parameter(torch.zeros([self.Nframe_m_obj, self.full_action_dim]))
            elif isinstance(self.mix_action, dict): # a config
                self.mix_action.mlp_spec[0] = self.mix_action.mlp_spec[0] * self.Nframe_m_obj
                self.mix_action.mlp_spec.append(self.Nframe_m_obj * self.full_action_dim)
                self.mix_action_params = build_model(self.mix_action)
            else:
                raise NotImplementedError(f"mix_action: {self.mix_action}")
        else:
            self.mix_action_params = None

        self.is_value = is_value
        if not self.is_value and not self.fuse_feature_single_action:
            # For action space decomposition, see https://github.com/haosulab/ManiSkill/wiki/Detailed-Explanation-of-Action
            self.final_mlp = ExtendedModuleList()

            body_mlp_cfg = deepcopy(mlp_cfg)
            if self.aux_regress in ['partial', 'full'] or self.mix_action:
                body_mlp_cfg.mlp_spec.append(self.full_action_dim)
            else:
                body_mlp_cfg.mlp_spec.append(self.base_action_dim)
            if self.action_hierarchy in ['h2b']:
                body_mlp_cfg.mlp_spec[0] += (self.Nframe_m_obj - 1) * self.per_hand_action_dim
            self.final_mlp.append(build_model(body_mlp_cfg))

            for _ in range(self.Nframe_m_obj - 1):
                hand_mlp_cfg = deepcopy(mlp_cfg)
                if self.aux_regress in ['full']:
                    hand_mlp_cfg.mlp_spec.append(self.base_action_dim + self.per_hand_action_dim)
                elif self.mix_action:
                    hand_mlp_cfg.mlp_spec.append(self.full_action_dim) 
                else:
                    hand_mlp_cfg.mlp_spec.append(self.per_hand_action_dim)
                if self.action_hierarchy in ['b2h']:
                    hand_mlp_cfg.mlp_spec[0] += self.base_action_dim
                self.final_mlp.append(build_model(hand_mlp_cfg))
        elif (not self.is_value) and self.fuse_feature_single_action:
            mlp_cfg.mlp_spec[0] *= self.Nframe_m_obj
            mlp_cfg.mlp_spec.append(self.full_action_dim)
            self.final_mlp = build_model(mlp_cfg)
        else:
            # for value network, we simply ignore aux_regress, action_hierarchy, and mix_action
            mlp_cfg.mlp_spec[0] = mlp_cfg.mlp_spec[0] * self.Nframe_m_obj
            self.final_mlp = build_model(mlp_cfg)

        self.saved_feature = None
        self.saved_visual_feature = None

        self.debug = debug

    def forward(
        self,
        obs,
        feature=None,
        visual_feature=None,
        save_feature=False,
        with_robot_state=True,
        **kwargs,
    ):
        assert isinstance(obs, dict), f"obs is not a dict! {type(obs)}"
        assert not (feature is not None and visual_feature is not None), f"You cannot provide visual_feature and feature at the same time!"
        self.saved_feature = None
        self.saved_visual_feature = None
        save_feature = save_feature or (feature is not None or visual_feature is not None)
        obs_keys = obs.keys()
        obs = copy(obs)

        assert "hand_pose" in obs_keys
        hand_pose = obs.pop("hand_pose") # [B, Nframe_m_obj - 1, 7]
        robot_state = None
        for key in ["state", "agent"]:
            if key in obs:
                assert robot_state is None, f"Please provide only one robot state! Obs Keys: {obs_keys}"
                robot_state = obs.pop(key)

        # Extract object frame information from observation
        if self.use_obj_frames is not None:
            obj_pose_info = get_obj_pose_info_from_obs(obs, self.use_obj_frames, use_rot=self.obj_frame_rot)
        else:
            obj_pose_info = None

        for key in obs_keys:
            if "_box" in key or "_seg" in key or key in ["point_to_inst_sem_label", "point_to_target_sem_label"]:
                obs.pop(key)
        if not ("xyz" in obs or "rgb" in obs or "rgbd" in obs):
            assert len(obs) == 1, f"Observations need to contain only one visual element! Obs Keys: {obs.keys()}!"
            obs = obs[list(obs.keys())[0]]

        if feature is None:
            if visual_feature is None:
                if not self.late_concat_state:
                    # hand_pose is used for transforming input point cloud into hand-centric point clouds
                    # obj_pose_info is used for transforming input point cloud into object-centric point clouds
                    feat = self.visual_nn(obs, hand_pose, robot_state=robot_state, obj_pose_info=obj_pose_info) 
                else:
                    feat = self.visual_nn(obs, hand_pose, robot_state=None, obj_pose_info=obj_pose_info)
                if obj_pose_info is not None:
                    feat = feat[:, :self.Nframe_m_obj, :] # obj_pose is only used for attention       
                # feat: [B, Nframe_m_obj, C]
            else:
                feat = visual_feature

            if self.late_concat_state:
                feat_cat = torch.cat([feat, robot_state[:, None, :].repeat_interleave(feat.size(1), dim=1)], dim=-1)
            else:
                feat_cat = feat

            if save_feature:
                self.saved_visual_feature = feat.clone()
                self.saved_feature = feat_cat.clone()
            
            if self.late_concat_state:
                feat = feat_cat
        else:
            feat = feature

        if self.is_value or ((not self.is_value) and self.fuse_feature_single_action):
            feat = feat.view([feat.size(0), -1])
            feat = self.final_mlp(feat)
            return feat
        else:
            if self.mix_action:
                if self.mix_action == True:
                    mix_action_params = self.mix_action_params[None, :, :]
                elif isinstance(self.mix_action, dict):
                    B, Nframe_m_obj, C = feat.size()
                    mix_action_params = self.mix_action_params(feat.view(B, -1))
                    mix_action_params = mix_action_params.view(B, Nframe_m_obj, -1) # the last dim equals the action dim
                if self.debug:
                    print(torch.softmax(mix_action_params[0], dim=0), flush=True)

            feat = feat.split(1, dim=1)
            feat = [x.squeeze(1) for x in feat]
            if self.action_hierarchy is None:
                for i in range(self.Nframe_m_obj):
                    feat[i] = self.final_mlp[i](feat[i])
            elif self.action_hierarchy in ['b2h']:
                feat[0] = self.final_mlp[0](feat[0])
                for i in range(1, self.Nframe_m_obj):
                    feat[i] = self.final_mlp[i](torch.cat([feat[i], feat[0]], dim=-1))
            elif self.action_hierarchy in ['h2b']:
                feat0_to_concat = [feat[0]]  
                for i in range(1, self.Nframe_m_obj):
                    feat[i] = self.final_mlp[i](feat[i])  
                    feat0_to_concat.append(feat[i])  
                feat[0] = self.final_mlp[0](torch.cat(feat0_to_concat, dim=-1))     

            if self.aux_regress is None:
                if self.mix_action:
                    feat = torch.stack(feat, dim=1)
                    feat = feat * torch.softmax(mix_action_params, dim=1)
                    feat = feat.sum(dim=1)
                else:
                    feat = torch.cat(feat, dim=-1) # action decomposition follows the original ordering of base and hand_pose
                return feat
            else:
                feat_ret, feat_hand_pred = feat[0][:, :self.base_action_dim], feat[0][:, self.base_action_dim:]
                if self.aux_regress in ['full']:
                    feat_base_pred = []
                for i in range(1, self.Nframe_m_obj):
                    if self.aux_regress in ['partial']:
                        feat_ret = torch.cat([feat_ret, feat[i]], dim=-1)
                    elif self.aux_regress in ['full']:
                        feat_ret = torch.cat([feat_ret, feat[i][:, self.base_action_dim:]], dim=-1)
                        feat_base_pred.append(feat[i][:, :self.base_action_dim])
                aux_loss = torch.mean((feat_hand_pred - feat_ret[:, self.base_action_dim:].detach()) ** 2)
                if self.aux_regress in ['full']:
                    feat_base_gt = feat_ret[:, :self.base_action_dim].detach()
                    for fbp in feat_base_pred:
                        aux_loss = aux_loss + torch.mean((fbp - feat_base_gt) ** 2)
                aux_loss = aux_loss * self.aux_regress_coeff
                return {'feat': feat_ret, 'aux_loss': aux_loss}






@BACKBONES.register_module()
class VisuomotorTransformerFrameLink(ExtendedModule):
    def __init__(self, visual_nn_cfg, final_mlp_cfg, link_visual_nn_cfg=None, is_value=False, visual_nn=None, freeze_visual_nn=False,
                 final_action_concat_state=True, residual_mlp_cfg=None, residual_sqex=False, residual_decompose_action_by_frame=False, 
                 use_obj_frames=None, obj_frame_rot=False, debug=False):
        """
        First, obtain visual features in different frames
            In the case of value network, features are concatenated to output value
            In the case of policy network, perform cross-attention between link embeddings and these visual features, along with
                self-attention between link embeddings; each link embedding outputs a dimension of the action space
        
        is_value: bool
            whether this is value network

        final_action_concat_state: whether to concat the robot state to the final mlp for action output

        residual_mlp_cfg: Config
            If not None, then for policy network, to improve training stability, add residual connections
        residual_sqex: bool
            If residual_mlp is not None, whether to use squeeze-excitation operation in the residual branch
        residual_decompose_action_by_frame: bool
            If use residual branch, then, for feature output at each frame and feature output after frame-to-frame attention, 
                whether, instead of concatenating features of different frames to output a single feature, add it through the residual
                branch to the final feature, such that these features predict the entire action space at once, 
                we let features at each frame predict the frame-specific part of action space

        use_obj_frames: None or a list of int
            If a list of int, this refers to the semantic id of obj frame to be selected to transform the point clouds, in addition
                to the hand frames used to transform the point clouds (this will result in 1+Nhand+len(use_obj_frames) frames)
        obj_frame_rot: bool
            If use_obj_frames is not None, then if obj_frame_rot, use object-centric rotation to transform the point clouds

        :return: [B, action_dim] action output before the final head (e.g. action mean before tanh)
        """                 
        super(VisuomotorTransformerFrameLink, self).__init__()
        self.use_obj_frames = use_obj_frames
        if self.use_obj_frames is not None and visual_nn_cfg is not None:
            assert isinstance(self.use_obj_frames, list) and isinstance(self.use_obj_frames[0], int)
            visual_nn_cfg['num_obj_frames'] = len(self.use_obj_frames)
        self.obj_frame_rot = obj_frame_rot

        self.visual_nn = build_model(visual_nn_cfg) if visual_nn is None else visual_nn

        self.link_visual_nn = build_model(link_visual_nn_cfg)

        self.Nframe = self.visual_nn.num_frames # all frames, including base, hand, and object frames
        if self.use_obj_frames is not None:
            self.Nframe_m_obj = self.Nframe - len(self.use_obj_frames) # base and hand frames
        else:
            self.Nframe_m_obj = self.Nframe
        self.base_action_dim = 4
        self.per_hand_action_dim = 9
        self.full_action_dim = self.base_action_dim + (self.Nframe_m_obj - 1) * self.per_hand_action_dim # dim of the entire action space        
        if freeze_visual_nn:
            get_logger().warning("We freeze the visual backbone!")
            freeze_params(self.visual_nn)  

        self.final_action_concat_state = final_action_concat_state

        self.is_value = is_value
        if self.is_value or link_visual_nn_cfg is None: # value network, or policy network with skipped link_visual_nn
            final_mlp_cfg.mlp_spec[0] = final_mlp_cfg.mlp_spec[0] * self.Nframe
            self.link_visual_nn = None
        else:
            assert final_mlp_cfg.mlp_spec[-1] == 1, "Action output should be 1 dim for each link"

        self.residual_decompose_action_by_frame = residual_decompose_action_by_frame
        if not self.is_value and residual_mlp_cfg is not None:
            residual_mlp_cfg1 = deepcopy(residual_mlp_cfg)
            residual_mlp_cfg2 = deepcopy(residual_mlp_cfg)
            residual_mlp_cfg2.mlp_spec[0] = residual_mlp_cfg2.mlp_spec[-1]
            if self.residual_decompose_action_by_frame:
                self.residual_mlps = ExtendedModuleList([
                    ExtendedModuleList([build_model(residual_mlp_cfg1) for _ in range(self.Nframe)]), # pre frame Transformer
                    ExtendedModuleList([build_model(residual_mlp_cfg1) for _ in range(self.Nframe)]), # after frame Transformer
                    build_model(residual_mlp_cfg2) # after Frame-link Transformer
                ])
            else:
                residual_mlp_cfg1.mlp_spec[0] = residual_mlp_cfg1.mlp_spec[0] * self.Nframe
                self.residual_mlps = ExtendedModuleList(
                    [build_model(residual_mlp_cfg1), build_model(residual_mlp_cfg1), build_model(residual_mlp_cfg2)]
                )
            if residual_sqex:
                out_channels = residual_mlp_cfg.mlp_spec[-1]
                self.residual_sqexs = ExtendedModuleList([
                    nn.Sequential(
                        nn.Linear(out_channels, out_channels // 10), nn.ReLU(), nn.Linear(out_channels // 10, out_channels), nn.Sigmoid()
                    ) for _ in range(2)
                ])
            else:
                self.residual_sqexs = None
        else:
            self.residual_mlps = None
            self.residual_sqexs = None

        self.final_mlp = build_model(final_mlp_cfg)            

        self.saved_feature = None
        self.saved_visual_feature = None

        self.debug = debug

    @staticmethod
    def cat_feat_state(feat, robot_state):
        # feat: [B, N, C1]; robot_state [B, C2]
        return torch.cat([feat, robot_state[:, None, :].repeat_interleave(feat.size(1), dim=1)], dim=-1)

    def forward(
        self,
        obs,
        feature=None,
        save_feature=False,
        with_robot_state=True,
        **kwargs,
    ):
        assert isinstance(obs, dict), f"obs is not a dict! {type(obs)}"
        self.saved_feature = None
        self.saved_visual_feature = None
        save_feature = save_feature or (feature is not None)
        obs_keys = obs.keys()
        obs = copy(obs)

        assert "hand_pose" in obs_keys
        hand_pose = obs.pop("hand_pose") # [B, Nframe-1, 7]
        robot_state = None
        for key in ["state", "agent"]:
            if key in obs:
                assert robot_state is None, f"Please provide only one robot state! Obs Keys: {obs_keys}"
                robot_state = obs.pop(key)

        # Extract object frame information from observation
        if self.use_obj_frames is not None:
            obj_pose_info = get_obj_pose_info_from_obs(obs, self.use_obj_frames, use_rot=self.obj_frame_rot)
        else:
            obj_pose_info = None

        for key in obs_keys:
            if "_box" in key or "_seg" in key or key in ["point_to_inst_sem_label", "point_to_target_sem_label"]:
                obs.pop(key)
        if not ("xyz" in obs or "rgb" in obs or "rgbd" in obs):
            assert len(obs) == 1, f"Observations need to contain only one visual element! Obs Keys: {obs.keys()}!"
            obs = obs[list(obs.keys())[0]]

        if feature is None:
            pre_tf_feat, tf_feat = self.visual_nn(obs, hand_pose, robot_state=None, obj_pose_info=obj_pose_info, return_pre_tf=True) 
            saved_feat = torch.stack([pre_tf_feat, tf_feat], dim=0)
            # hand_pose is used for transforming input point cloud into hand-centric point clouds
            # pre_tf_feat, feat: [B, Nframe, C]

            if save_feature:
                self.saved_visual_feature = self.cat_feat_state(tf_feat, robot_state).clone()
                self.saved_feature = saved_feat.clone()
        else:
            assert feature.shape[0] == 2
            pre_tf_feat, tf_feat = feature[0], feature[1] # deserialize the saved feature

        if self.is_value or self.link_visual_nn is None:
            feat = self.cat_feat_state(tf_feat, robot_state)
            feat = feat.view([feat.size(0), -1])
            feat = self.final_mlp(feat)
            return feat
        else:
            link_visual_feat = self.link_visual_nn(tf_feat, robot_state=robot_state) # [B, dim_action, C]

            if self.residual_mlps is not None:
                pre_tf_feat_residual = self.cat_feat_state(pre_tf_feat, robot_state) # [B, Nframe, C]           
                tf_feat_residual = self.cat_feat_state(tf_feat, robot_state)
                if self.residual_decompose_action_by_frame:
                    pre_tf_feat_residual = [self.residual_mlps[0][i](pre_tf_feat_residual[:, i, :]) for i in range(self.Nframe)]
                    pre_tf_feat_residual = torch.stack(pre_tf_feat_residual, dim=1) # [B, Nframe, C]    
                    tf_feat_residual = [self.residual_mlps[1][i](tf_feat_residual[:, i, :]) for i in range(self.Nframe)]
                    tf_feat_residual = torch.stack(tf_feat_residual, dim=1) # [B, Nframe, C]  
                else:
                    pre_tf_feat_residual = pre_tf_feat_residual.view(pre_tf_feat_residual.size(0), -1)
                    pre_tf_feat_residual = self.residual_mlps[0](pre_tf_feat_residual) # [B, C]    
                    tf_feat_residual = tf_feat_residual.view(tf_feat_residual.size(0), -1)
                    tf_feat_residual = self.residual_mlps[1](tf_feat_residual) # [B, C]
                link_visual_feat_residual = self.residual_mlps[2](link_visual_feat) # [B, dim_action, C]
                if self.residual_sqexs is not None:
                    channel_excitation = self.residual_sqexs[0](tf_feat_residual)
                    # print(channel_excitation[0][:10], flush=True)
                    tf_feat_residual = tf_feat_residual * channel_excitation
                    channel_excitation = self.residual_sqexs[1](torch.mean(link_visual_feat_residual, dim=1))
                    # print(channel_excitation[0][:10], flush=True)
                    link_visual_feat_residual = link_visual_feat_residual * channel_excitation[:, None, :]
                if self.residual_decompose_action_by_frame:
                    cur_idx = 0
                    feats = []
                    for i in range(self.Nframe_m_obj):
                        if i == 0:
                            feats.append(
                                link_visual_feat_residual[:, cur_idx:cur_idx+self.base_action_dim, :]
                                + pre_tf_feat_residual[:, [i], :]
                                + tf_feat_residual[:, [i], :]
                            )
                            cur_idx += self.base_action_dim
                        else:
                            feats.append(
                                link_visual_feat_residual[:, cur_idx:cur_idx+self.per_hand_action_dim, :]
                                + pre_tf_feat_residual[:, [i], :]
                                + tf_feat_residual[:, [i], :]
                            )                            
                            cur_idx += self.per_hand_action_dim
                    feat = torch.cat(feats, dim=1)
                    # if self.use_obj_frames is not None: # commenting this out -> obj frame feature only used for attention, not for action prediction
                    #     for i in range(self.Nframe_m_obj, self.Nframe):
                    #         feat = feat + pre_tf_feat_residual[:, [i], :] + tf_feat_residual[:, [i], :]
                else:
                    feat = pre_tf_feat_residual[:, None, :] + tf_feat_residual[:, None, :] + link_visual_feat_residual
            else:
                feat = link_visual_feat

            if self.final_action_concat_state:
                feat = self.cat_feat_state(feat, robot_state)
            feat = self.final_mlp(feat)
            return feat.squeeze(1) # [B, dim_action]