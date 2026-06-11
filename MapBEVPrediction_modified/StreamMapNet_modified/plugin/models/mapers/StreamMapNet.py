import mmcv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torchvision.models.resnet import resnet18, resnet50

from mmdet3d.models.builder import (build_backbone, build_head,
                                    build_neck)

from .base_mapper import BaseMapper, MAPPERS
from copy import deepcopy
from ..utils.memory_buffer import StreamTensorMemory
from mmcv.cnn.utils import constant_init, kaiming_init
from .branch_protocol import (
    BRANCH_MODE_CAMERA_ONLY,
    BRANCH_MODE_DROP_CAMERA,
    BRANCH_MODE_DROP_SATELLITE,
    BRANCH_MODE_FUSION,
    BRANCH_MODE_SATELLITE_ONLY,
    resolve_branch_mode,
    uses_aerial_encoder,
    uses_aerial_fuser,
    uses_camera_encoder,
)

@MAPPERS.register_module()
class StreamMapNet(BaseMapper):

    def __init__(self,
                 bev_h,
                 bev_w,
                 roi_size,
                 backbone_cfg=dict(),
                 head_cfg=dict(),
                 neck_cfg=None,
                 model_name=None, 
                 streaming_cfg=dict(),
                 pretrained=None,
                 use_AID4AD=False,
                 AID4AD_only=False,
                 branch_mode=None,
                 return_branch_features=False,
                 distill_cfg=None,
                 probe_cfg=None,
                 **kwargs):
        super().__init__()

        #Attribute
        self.model_name = model_name
        self.last_epoch = None
  
        self.backbone = build_backbone(backbone_cfg)

        self.branch_mode = resolve_branch_mode(use_AID4AD, AID4AD_only, branch_mode)
        self.return_branch_features = return_branch_features
        self.distill_cfg = distill_cfg or {}
        self.probe_cfg = probe_cfg or {}
        self.distill_feature = self.distill_cfg.get('feature_name', 'camera_branch_feature')
        self.distill_loss = self.distill_cfg.get('loss_name', 'mse')
        self.distill_weight = self.distill_cfg.get('loss_weight', 0.0)
        self.distill_use_mask = self.distill_cfg.get('use_mask', False)
        self.distill_mask_thickness = self.distill_cfg.get('mask_thickness', 3)
        self.teacher_branch_mode = self.distill_cfg.get('teacher_branch_mode', BRANCH_MODE_CAMERA_ONLY)
        self.teacher_ckpt = self.distill_cfg.get('teacher_ckpt')
        self.teacher_model = None

        self.use_AID4AD = use_AID4AD or self.branch_mode in (BRANCH_MODE_FUSION, BRANCH_MODE_SATELLITE_ONLY)
        self.AID4AD_only = AID4AD_only

        if not self.use_AID4AD and self.AID4AD_only:
            raise ValueError('AID4AD_only should be used with use_AID4AD=True')

        if self.use_AID4AD:
            from .resunet import ResNetUNet, ConvFuser, DownsampleCNN

            bev_embed_dims = backbone_cfg.transformer.embed_dims
            self.aid_encoder = ResNetUNet(outC=64)
            self.aid_downsampler = DownsampleCNN(in_channels=64, hidden_dim=128)
            self.aid_fuser = ConvFuser(2*bev_embed_dims, bev_embed_dims)


        if neck_cfg is not None:
            self.neck = build_head(neck_cfg)
        else:
            self.neck = nn.Identity()

        self.head = build_head(head_cfg)
        self.num_decoder_layers = self.head.transformer.decoder.num_layers
        
        # BEV 
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.roi_size = roi_size

        if streaming_cfg:
            self.streaming_bev = streaming_cfg['streaming_bev']
        else:
            self.streaming_bev = False
        if self.streaming_bev:
            self.stream_fusion_neck = build_neck(streaming_cfg['fusion_cfg'])
            self.batch_size = streaming_cfg['batch_size']
            self.bev_memory = StreamTensorMemory(
                self.batch_size,
            )
            
            xmin, xmax = -roi_size[0]/2, roi_size[0]/2
            ymin, ymax = -roi_size[1]/2, roi_size[1]/2
            x = torch.linspace(xmin, xmax, bev_w)
            y = torch.linspace(ymax, ymin, bev_h)
            y, x = torch.meshgrid(y, x)
            z = torch.zeros_like(x)
            ones = torch.ones_like(x)
            plane = torch.stack([x, y, z, ones], dim=-1)

            self.register_buffer('plane', plane.double())
        
        self.init_weights(pretrained)
        if self.teacher_ckpt:
            self.teacher_model = self._build_teacher_model(
                bev_h=bev_h,
                bev_w=bev_w,
                roi_size=roi_size,
                backbone_cfg=backbone_cfg,
                head_cfg=head_cfg,
                neck_cfg=neck_cfg,
                model_name=model_name,
                streaming_cfg=streaming_cfg,
                use_AID4AD=use_AID4AD,
                AID4AD_only=False,
            )

    def init_weights(self, pretrained=None):
        """Initialize model weights."""
        if pretrained:
            import logging
            logger = logging.getLogger()
            from mmcv.runner import load_checkpoint
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        else:
            try:
                self.neck.init_weights()
            except AttributeError:
                pass
            if self.streaming_bev:
                self.stream_fusion_neck.init_weights()

    def _build_teacher_model(self, bev_h, bev_w, roi_size, backbone_cfg, head_cfg, neck_cfg, model_name, streaming_cfg, use_AID4AD, AID4AD_only):
        from mmcv.runner import load_checkpoint

        teacher_model = type(self)(
            bev_h=bev_h,
            bev_w=bev_w,
            roi_size=roi_size,
            backbone_cfg=deepcopy(backbone_cfg),
            head_cfg=deepcopy(head_cfg),
            neck_cfg=deepcopy(neck_cfg),
            model_name=model_name,
            streaming_cfg=deepcopy(streaming_cfg),
            pretrained=None,
            use_AID4AD=use_AID4AD,
            AID4AD_only=AID4AD_only,
            branch_mode=self.teacher_branch_mode,
            return_branch_features=True,
            distill_cfg=None,
        )
        load_checkpoint(teacher_model, self.teacher_ckpt, strict=False, logger=None)
        teacher_model.eval()
        for parameter in teacher_model.parameters():
            parameter.requires_grad = False
        return teacher_model

    def _satellite_feature_channels(self):
        if not self.use_AID4AD:
            return 0
        return self.aid_downsampler.conv[-2].out_channels

    def _zero_satellite_feature(self, reference_feature):
        channels = self._satellite_feature_channels()
        return torch.zeros(
            reference_feature.size(0),
            channels,
            reference_feature.size(2),
            reference_feature.size(3),
            device=reference_feature.device,
            dtype=reference_feature.dtype,
        )

    def _zero_camera_feature(self, reference_feature):
        return torch.zeros_like(reference_feature)

    def extract_branch_features(self, img, aerial_img=None, points=None, img_metas=None, branch_mode=None):
        branch_mode = branch_mode or self.branch_mode

        camera_branch_feature = None
        if uses_camera_encoder(branch_mode):
            camera_branch_feature = self.backbone(img, img_metas=img_metas, points=points)

        satellite_branch_feature = None
        if uses_aerial_encoder(branch_mode):
            satellite_branch_feature = self.aid_encoder(aerial_img)
            satellite_branch_feature = self.aid_downsampler(satellite_branch_feature)

        if branch_mode == BRANCH_MODE_SATELLITE_ONLY:
            fusion_feature = satellite_branch_feature
        elif branch_mode == BRANCH_MODE_CAMERA_ONLY:
            fusion_feature = camera_branch_feature
        elif branch_mode == BRANCH_MODE_DROP_SATELLITE:
            satellite_branch_feature = self._zero_satellite_feature(camera_branch_feature)
            fusion_feature = self.aid_fuser(
                torch.cat([camera_branch_feature, satellite_branch_feature], dim=1))
        elif branch_mode == BRANCH_MODE_DROP_CAMERA:
            camera_branch_feature = self._zero_camera_feature(satellite_branch_feature)
            fusion_feature = self.aid_fuser(
                torch.cat([camera_branch_feature, satellite_branch_feature], dim=1))
        elif uses_aerial_fuser(branch_mode):
            fusion_feature = self.aid_fuser(
                torch.cat([camera_branch_feature, satellite_branch_feature], dim=1))
        else:
            fusion_feature = camera_branch_feature

        return {
            'branch_mode': branch_mode,
            'camera_branch_feature': camera_branch_feature,
            'satellite_branch_feature': satellite_branch_feature,
            'fusion_feature': fusion_feature,
        }

    def get_probe_decoder(self, decoder_type):
        if decoder_type not in ('camera', 'satellite', 'head'):
            raise ValueError(f'Unsupported probe decoder type: {decoder_type}')
        return self.head

    def get_probe_encoder_modules(self, decoder_type):
        if decoder_type == 'camera':
            modules = [self.backbone]
            if self.streaming_bev:
                modules.append(self.stream_fusion_neck)
            modules.extend([self.neck])
            if self.use_AID4AD:
                modules.extend([self.aid_encoder, self.aid_downsampler, self.aid_fuser])
            return modules
        if decoder_type == 'satellite':
            modules = [self.aid_encoder, self.aid_downsampler]
            if self.use_AID4AD:
                modules.extend([self.aid_fuser])
            if self.streaming_bev:
                modules.append(self.stream_fusion_neck)
            modules.extend([self.neck, self.backbone])
            return modules
        raise ValueError(f'Unsupported probe decoder type: {decoder_type}')

    def _rasterize_vector_line(self, foreground_mask, line):
        if line.ndim == 3:
            line = line[0]
        if line.shape[0] < 2:
            return

        height, width = foreground_mask.shape
        line = line[:, :2].to(device=foreground_mask.device, dtype=torch.float32)
        line = line.clamp(0.0, 1.0)
        coords = torch.stack([
            line[:, 0] * (width - 1),
            line[:, 1] * (height - 1),
        ], dim=-1)

        radius = max(int(self.distill_mask_thickness) // 2, 0)
        for start, end in zip(coords[:-1], coords[1:]):
            delta = end - start
            steps = int(torch.ceil(delta.abs().max()).item()) + 1
            if steps <= 1:
                samples = start.view(1, 2)
            else:
                t = torch.linspace(0, 1, steps, device=foreground_mask.device)
                samples = start.view(1, 2) * (1 - t).view(-1, 1) + end.view(1, 2) * t.view(-1, 1)

            xs = samples[:, 0].round().long().clamp(0, width - 1)
            ys = samples[:, 1].round().long().clamp(0, height - 1)
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    px = (xs + dx).clamp(0, width - 1)
                    py = (ys + dy).clamp(0, height - 1)
                    foreground_mask[py, px] = True

    def _build_distill_non_mask(self, vectors, spatial_size, device):
        """Match Satfor KD: distill where GT map foreground is absent."""
        height, width = spatial_size
        foreground = torch.zeros((len(vectors), height, width), dtype=torch.bool, device=device)
        for batch_idx, vector_by_label in enumerate(vectors):
            for _label, lines in vector_by_label.items():
                for line in lines:
                    self._rasterize_vector_line(
                        foreground[batch_idx],
                        torch.as_tensor(line),
                    )
        return ~foreground

    def _compute_distillation_loss(self, student_feature, teacher_feature, non_mask=None):
        teacher_feature = teacher_feature.detach()
        scale = teacher_feature.pow(2).mean().sqrt().clamp_min(1e-6)
        student_norm = student_feature / scale
        teacher_norm = teacher_feature / scale
        if non_mask is not None:
            if non_mask.shape[-2:] != student_norm.shape[-2:]:
                non_mask = F.interpolate(
                    non_mask.float().unsqueeze(1),
                    size=student_norm.shape[-2:],
                    mode='nearest',
                ).squeeze(1).bool()
            student_norm = student_norm.permute(0, 2, 3, 1)[non_mask]
            teacher_norm = teacher_norm.permute(0, 2, 3, 1)[non_mask]
            if student_norm.numel() == 0:
                return student_feature.new_zeros(())
        if self.distill_loss == 'l1':
            return F.l1_loss(student_norm, teacher_norm)
        return F.mse_loss(student_norm, teacher_norm)

    def update_bev_feature(self, curr_bev_feats, img_metas):
        '''
        Args:
            curr_bev_feat: torch.Tensor of shape [B, neck_input_channels, H, W]
            img_metas: current image metas (List of #bs samples)
            bev_memory: where to load and store (training and testing use different buffer)
            pose_memory: where to load and store (training and testing use different buffer)

        Out:
            fused_bev_feat: torch.Tensor of shape [B, neck_input_channels, H, W]
        '''

        bs = curr_bev_feats.size(0)
        fused_feats_list = []

        memory = self.bev_memory.get(img_metas)
        bev_memory, pose_memory = memory['tensor'], memory['img_metas']
        is_first_frame_list = memory['is_first_frame']

        for i in range(bs):
            is_first_frame = is_first_frame_list[i]
            if is_first_frame:
                new_feat = self.stream_fusion_neck(curr_bev_feats[i].clone().detach(), curr_bev_feats[i])
                fused_feats_list.append(new_feat)
            else:
                # else, warp buffered bev feature to current pose
                prev_e2g_trans = self.plane.new_tensor(pose_memory[i]['ego2global_translation'], dtype=torch.float64)
                prev_e2g_rot = self.plane.new_tensor(pose_memory[i]['ego2global_rotation'], dtype=torch.float64)
                curr_e2g_trans = self.plane.new_tensor(img_metas[i]['ego2global_translation'], dtype=torch.float64)
                curr_e2g_rot = self.plane.new_tensor(img_metas[i]['ego2global_rotation'], dtype=torch.float64)
                
                prev_g2e_matrix = torch.eye(4, dtype=torch.float64, device=prev_e2g_trans.device)
                prev_g2e_matrix[:3, :3] = prev_e2g_rot.T
                prev_g2e_matrix[:3, 3] = -(prev_e2g_rot.T @ prev_e2g_trans)

                curr_e2g_matrix = torch.eye(4, dtype=torch.float64, device=prev_e2g_trans.device)
                curr_e2g_matrix[:3, :3] = curr_e2g_rot
                curr_e2g_matrix[:3, 3] = curr_e2g_trans

                curr2prev_matrix = prev_g2e_matrix @ curr_e2g_matrix
                prev_coord = torch.einsum('lk,ijk->ijl', curr2prev_matrix, self.plane).float()[..., :2]

                # from (-30, 30) or (-15, 15) to (-1, 1)
                prev_coord[..., 0] = prev_coord[..., 0] / (self.roi_size[0]/2)
                prev_coord[..., 1] = -prev_coord[..., 1] / (self.roi_size[1]/2)

                warped_feat = F.grid_sample(bev_memory[i].unsqueeze(0), 
                                prev_coord.unsqueeze(0), 
                                padding_mode='zeros', align_corners=False).squeeze(0)
                new_feat = self.stream_fusion_neck(warped_feat, curr_bev_feats[i])
                fused_feats_list.append(new_feat)

        fused_feats = torch.stack(fused_feats_list, dim=0)

        self.bev_memory.update(fused_feats, img_metas)
        
        return fused_feats

    def forward_train(self, img, vectors, aerial_img=None, points=None, img_metas=None, **kwargs):
        '''
        Args:
            img: torch.Tensor of shape [B, N, 3, H, W]
                N: number of cams
            vectors: list[list[Tuple(lines, length, label)]]
                - lines: np.array of shape [num_points, 2]. 
                - length: int
                - label: int
                len(vectors) = batch_size
                len(vectors[_b]) = num of lines in sample _b
            img_metas: 
                img_metas['lidar2img']: [B, N, 4, 4]
        Out:
            loss, log_vars, num_sample
        '''
        #  prepare labels and images

        gts, img, img_metas, valid_idx, points = self.batch_data(
            vectors, img, img_metas, img.device, points)
        
        bs = img.shape[0]

        # Backbone
        branch_features = self.extract_branch_features(
            img,
            aerial_img=aerial_img,
            points=points,
            img_metas=img_metas,
        )
        self.latest_branch_features = branch_features
        _bev_feats = branch_features['fusion_feature']
        
        if self.streaming_bev:
            self.bev_memory.train()
            _bev_feats = self.update_bev_feature(_bev_feats, img_metas)
        
        # Neck
        bev_feats = self.neck(_bev_feats)

        preds_list, loss_dict, det_match_idxs, det_match_gt_idxs = self.head(
            bev_features=bev_feats, 
            img_metas=img_metas, 
            gts=gts,
            return_loss=True)
        
        # format loss
        loss = 0
        for name, var in loss_dict.items():
            loss = loss + var

        if self.teacher_model is not None and branch_features.get(self.distill_feature) is not None:
            with torch.no_grad():
                teacher_branch_features = self.teacher_model.extract_branch_features(
                    img,
                    aerial_img=aerial_img,
                    points=points,
                    img_metas=img_metas,
                    branch_mode=self.teacher_branch_mode,
                )
            non_mask = None
            if self.distill_use_mask:
                non_mask = self._build_distill_non_mask(
                    vectors,
                    branch_features[self.distill_feature].shape[-2:],
                    branch_features[self.distill_feature].device,
                )
            distill_loss = self._compute_distillation_loss(
                branch_features[self.distill_feature],
                teacher_branch_features[self.distill_feature],
                non_mask=non_mask,
            )
            distill_term = distill_loss * self.distill_weight
            loss = loss + distill_term
            loss_dict['distill_loss'] = distill_term

        # update the log
        log_vars = {k: v.item() for k, v in loss_dict.items()}
        log_vars.update({'total': loss.item()})

        num_sample = img.size(0)

        return loss, log_vars, num_sample

    @torch.no_grad()
    def forward_test(self, img, aerial_img=None, points=None, img_metas=None, **kwargs):
        '''
            inference pipeline
        '''

        #  prepare labels and images
        
        tokens = []
        for img_meta in img_metas:
            tokens.append(img_meta['token'])

        branch_features = self.extract_branch_features(
            img,
            aerial_img=aerial_img,
            points=points,
            img_metas=img_metas,
        )
        self.latest_branch_features = branch_features
        _bev_feats = branch_features['fusion_feature']

        img_shape = [_bev_feats.shape[2:] for i in range(_bev_feats.shape[0])]

        if self.streaming_bev:
            self.bev_memory.eval()
            _bev_feats = self.update_bev_feature(_bev_feats, img_metas)
            
        # Neck
        bev_feats = self.neck(_bev_feats)

        preds_list = self.head(bev_feats, img_metas=img_metas, return_loss=False)
        
        # take predictions from the last layer
        preds_dict = preds_list[-1]

        results_list = self.head.post_process(preds_dict, tokens)

        return results_list

    def batch_data(self, vectors, imgs, img_metas, device, points=None):
        bs = len(vectors)
        # filter none vector's case
        num_gts = []
        for idx in range(bs):
            num_gts.append(sum([len(v) for k, v in vectors[idx].items()]))
        valid_idx = [i for i in range(bs) if num_gts[i] > 0]
        assert len(valid_idx) == bs # make sure every sample has gts

        gts = []
        all_labels_list = []
        all_lines_list = []
        for idx in range(bs):
            labels = []
            lines = []
            for label, _lines in vectors[idx].items():
                for _line in _lines:
                    labels.append(label)
                    if len(_line.shape) == 3: # permutation
                        num_permute, num_points, coords_dim = _line.shape
                        lines.append(torch.tensor(_line).reshape(num_permute, -1)) # (38, 40)
                    elif len(_line.shape) == 2:
                        lines.append(torch.tensor(_line).reshape(-1)) # (40, )
                    else:
                        assert False

            all_labels_list.append(torch.tensor(labels, dtype=torch.long).to(device))
            all_lines_list.append(torch.stack(lines).float().to(device))

        gts = {
            'labels': all_labels_list,
            'lines': all_lines_list
        }
        
        gts = [deepcopy(gts) for _ in range(self.num_decoder_layers)]

        return gts, imgs, img_metas, valid_idx, points

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        if self.streaming_bev:
            self.bev_memory.train(*args, **kwargs)
        if self.teacher_model is not None:
            self.teacher_model.eval()
    
    def eval(self):
        super().eval()
        if self.streaming_bev:
            self.bev_memory.eval()
        if self.teacher_model is not None:
            self.teacher_model.eval()

