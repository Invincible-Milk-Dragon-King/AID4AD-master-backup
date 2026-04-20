import torch
from torch import nn

from data.utils import gen_dx_bx

from .base import BevEncode, CamEncode
from .homography import IPM


class HDMapNetCameraBaseline(nn.Module):
    def __init__(self, data_conf, args, instance_seg=True, embedded_dim=16, direction_pred=True, direction_dim=36):
        super().__init__()
        self.camC = 64
        self.downsample = 16
        self.branch_mode = getattr(args, 'branch_mode', 'camera_only')
        self.return_branch_features = getattr(args, 'return_branch_features', False)
        _, _, nx = gen_dx_bx(data_conf['xbound'], data_conf['ybound'], data_conf['zbound'])
        final_H, final_W = nx[1].item(), nx[0].item()
        fv_size = (data_conf['image_size'][0] // self.downsample, data_conf['image_size'][1] // self.downsample)
        bv_size = (final_H // 5, final_W // 5)

        self.camencode = CamEncode(self.camC)
        self.view_fusion = ViewTransformation(fv_size=fv_size, bv_size=bv_size)

        res_x = bv_size[1] * 3 // 4
        ipm_xbound = [-res_x, res_x, 4 * res_x / final_W]
        ipm_ybound = [-res_x / 2, res_x / 2, 2 * res_x / final_H]
        self.ipm = IPM(args, ipm_xbound, ipm_ybound, N=6, C=self.camC, extrinsic=True)
        self.up_sampler = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.camera_bevencode = BevEncode(
            inC=self.camC,
            outC=data_conf['num_channels'],
            instance_seg=instance_seg,
            embedded_dim=embedded_dim,
            direction_pred=direction_pred,
            direction_dim=direction_dim + 1,
        )

    def get_Ks_RTs_and_post_RTs(self, intrins, rots, trans, post_rots, post_trans):
        B, N, _, _ = intrins.shape
        Ks = intrins.new_zeros((B, N, 4, 4))
        Ks[:, :, 0, 0] = 1
        Ks[:, :, 1, 1] = 1
        Ks[:, :, 2, 2] = 1
        Ks[:, :, 3, 3] = 1
        Ks[:, :, :3, :3] = intrins

        Rs = rots.new_zeros((B, N, 4, 4))
        Rs[:, :, 0, 0] = 1
        Rs[:, :, 1, 1] = 1
        Rs[:, :, 2, 2] = 1
        Rs[:, :, 3, 3] = 1
        Rs[:, :, :3, :3] = rots.transpose(-1, -2).contiguous()

        Ts = trans.new_zeros((B, N, 4, 4))
        Ts[:, :, 0, 0] = 1
        Ts[:, :, 1, 1] = 1
        Ts[:, :, 2, 2] = 1
        Ts[:, :, 3, 3] = 1
        Ts[:, :, :3, 3] = -trans
        RTs = Rs @ Ts

        return Ks, RTs, None

    def get_cam_feats(self, img):
        B, N, C, imH, imW = img.shape
        img = img.view(B * N, C, imH, imW)
        img = self.camencode(img)
        return img.view(B, N, self.camC, imH // self.downsample, imW // self.downsample)

    def encode_camera_branch(self, img, trans, rots, intrins, post_trans, post_rots, car_trans, yaw_pitch_roll):
        camera_features = self.get_cam_feats(img)
        camera_features = self.view_fusion(camera_features)
        Ks, RTs, post_RTs = self.get_Ks_RTs_and_post_RTs(
            intrins, rots, trans, post_rots, post_trans
        )
        topdown = self.ipm(camera_features, Ks, RTs, car_trans, yaw_pitch_roll, post_RTs)
        return self.up_sampler(topdown)

    def forward(
        self,
        img,
        trans,
        rots,
        intrins,
        post_trans,
        post_rots,
        lidar_data,
        lidar_mask,
        car_trans,
        yaw_pitch_roll,
        map_prior,
        branch_mode=None,
        return_branch_features=None,
    ):
        if return_branch_features is None:
            return_branch_features = self.return_branch_features

        camera_branch_feature = self.encode_camera_branch(
            img, trans, rots, intrins, post_trans, post_rots, car_trans, yaw_pitch_roll
        )
        semantic, embedding, direction = self.camera_bevencode(camera_branch_feature)

        if return_branch_features:
            return {
                'semantic': semantic,
                'embedding': embedding,
                'direction': direction,
                'branch_mode': branch_mode or self.branch_mode,
                'camera_branch_feature': camera_branch_feature,
                'satellite_branch_feature': None,
                'fusion_feature': camera_branch_feature,
            }
        return semantic, embedding, direction

    def get_probe_decoder(self, decoder_type: str):
        if decoder_type != 'camera':
            raise ValueError(f"Unsupported probe decoder type: {decoder_type}")
        return self.camera_bevencode

    def get_probe_encoder_modules(self, decoder_type: str):
        if decoder_type != 'camera':
            raise ValueError(f"Unsupported probe decoder type: {decoder_type}")
        return [self.camencode, self.view_fusion, self.ipm, self.up_sampler]


class ViewTransformation(nn.Module):
    def __init__(self, fv_size, bv_size, n_views=6):
        super().__init__()
        self.n_views = n_views
        self.bv_size = bv_size
        fv_dim = fv_size[0] * fv_size[1]
        bv_dim = bv_size[0] * bv_size[1]
        self.hw_mat = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fv_dim, bv_dim),
                nn.ReLU(),
                nn.Linear(bv_dim, bv_dim),
                nn.ReLU(),
            )
            for _ in range(self.n_views)
        ])

    def forward(self, feat):
        B, N, C, H, W = feat.shape
        feat = feat.view(B, N, C, H * W)
        outputs = []
        for i in range(N):
            output = self.hw_mat[i](feat[:, i]).view(B, C, self.bv_size[0], self.bv_size[1])
            outputs.append(output)
        return torch.stack(outputs, 1)
