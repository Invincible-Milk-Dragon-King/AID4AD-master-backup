import torch
from torch import nn

from .homography import bilinear_sampler, IPM
from .utils import plane_grid_2d, get_rot_2d, cam_to_pixel
from .pointpillar import PointPillarEncoder
from .base import CamEncode, BevEncode
from data.utils import gen_dx_bx


class ViewTransformation(nn.Module):
    def __init__(self, fv_size, bv_size, n_views=6):
        super(ViewTransformation, self).__init__()
        self.n_views = n_views
        self.hw_mat = []
        self.bv_size = bv_size
        fv_dim = fv_size[0] * fv_size[1]
        bv_dim = bv_size[0] * bv_size[1]
        for i in range(self.n_views):
            fc_transform = nn.Sequential(
                nn.Linear(fv_dim, bv_dim),
                nn.ReLU(),
                nn.Linear(bv_dim, bv_dim),
                nn.ReLU()
            )
            self.hw_mat.append(fc_transform)
        self.hw_mat = nn.ModuleList(self.hw_mat)

    def forward(self, feat):
        B, N, C, H, W = feat.shape
        feat = feat.view(B, N, C, H*W)
        outputs = []
        for i in range(N):
            output = self.hw_mat[i](feat[:, i]).view(B, C, self.bv_size[0], self.bv_size[1])
            outputs.append(output)
        outputs = torch.stack(outputs, 1)
        return outputs


class HDMapNet(nn.Module):
    def __init__(self, data_conf, instance_seg=True, embedded_dim=16, direction_pred=True, direction_dim=36, lidar=False):
        super(HDMapNet, self).__init__()
        self.camC = 64
        self.lidarC = 128
        self.downsample = 16

        dx, bx, nx = gen_dx_bx(data_conf['xbound'], data_conf['ybound'], data_conf['zbound'])
        final_H, final_W = nx[1].item(), nx[0].item()

        self.camencode = CamEncode(self.camC)
        fv_size = (data_conf['image_size'][0]//self.downsample, data_conf['image_size'][1]//self.downsample)
        bv_size = (final_H//5, final_W//5)
        self.view_fusion = ViewTransformation(fv_size=fv_size, bv_size=bv_size)

        res_x = bv_size[1] * 3 // 4
        ipm_xbound = [-res_x, res_x, 4*res_x/final_W]
        ipm_ybound = [-res_x/2, res_x/2, 2*res_x/final_H]
        self.ipm = IPM(ipm_xbound, ipm_ybound, N=6, C=self.camC, extrinsic=True)
        self.up_sampler = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.lidar = lidar
        # Always keep a camera-only decoder for Probe(C→C) / Probe(F→C).
        self.camera_bevencode = BevEncode(
            inC=self.camC,
            outC=data_conf['num_channels'],
            instance_seg=instance_seg,
            embedded_dim=embedded_dim,
            direction_pred=direction_pred,
            direction_dim=direction_dim+1,
        )
        if lidar:
            self.pp = PointPillarEncoder(
                self.lidarC, data_conf['xbound'], data_conf['ybound'], data_conf['zbound']
            )
            # LiDAR-only decoder for Probe(L→L) / Probe(F→L) / Full-L (protocol 1).
            self.lidar_bevencode = BevEncode(
                inC=self.lidarC,
                outC=data_conf['num_channels'],
                instance_seg=instance_seg,
                embedded_dim=embedded_dim,
                direction_pred=direction_pred,
                direction_dim=direction_dim+1,
            )
            # Fusion decoder (cam+lidar channels). Kept for Full-F compatibility.
            self.bevencode = BevEncode(
                inC=self.camC+self.lidarC,
                outC=data_conf['num_channels'],
                instance_seg=instance_seg,
                embedded_dim=embedded_dim,
                direction_pred=direction_pred,
                direction_dim=direction_dim+1,
            )
        else:
            # Camera-only model: share weights with camera_bevencode so old
            # `bevencode.*` checkpoints still load.
            self.bevencode = self.camera_bevencode

    def get_Ks_RTs_and_post_RTs(self, intrins, rots, trans, post_rots, post_trans):
        B, N, _, _ = intrins.shape
        Ks = torch.eye(4, device=intrins.device).view(1, 1, 4, 4).repeat(B, N, 1, 1)

        Rs = torch.eye(4, device=rots.device).view(1, 1, 4, 4).repeat(B, N, 1, 1)
        Rs[:, :, :3, :3] = rots.transpose(-1, -2).contiguous()
        Ts = torch.eye(4, device=trans.device).view(1, 1, 4, 4).repeat(B, N, 1, 1)
        Ts[:, :, :3, 3] = -trans
        RTs = Rs @ Ts

        post_RTs = None

        return Ks, RTs, post_RTs

    def get_cam_feats(self, x):
        B, N, C, imH, imW = x.shape
        x = x.view(B*N, C, imH, imW)
        x = self.camencode(x)
        x = x.view(B, N, self.camC, imH//self.downsample, imW//self.downsample)
        return x

    def encode_camera_topdown(self, img, trans, rots, intrins, post_trans, post_rots, car_trans, yaw_pitch_roll):
        x = self.get_cam_feats(img)
        x = self.view_fusion(x)
        Ks, RTs, post_RTs = self.get_Ks_RTs_and_post_RTs(intrins, rots, trans, post_rots, post_trans)
        topdown = self.ipm(x, Ks, RTs, car_trans, yaw_pitch_roll, post_RTs)
        return self.up_sampler(topdown)

    def get_probe_decoder(self, decoder_type):
        if decoder_type == "camera":
            return self.camera_bevencode
        if decoder_type == "lidar":
            if not self.lidar:
                raise ValueError("lidar probe decoder requires HDMapNet with lidar=True")
            return self.lidar_bevencode
        raise ValueError(f"Unsupported probe decoder type: {decoder_type}")

    def get_probe_encoder_modules(self, decoder_type):
        if decoder_type == "camera":
            # LiDAR encoder is not part of the camera branch under probe.
            return [self.camencode, self.view_fusion, self.ipm, self.up_sampler]
        if decoder_type == "lidar":
            if not self.lidar:
                raise ValueError("lidar probe encoder requires HDMapNet with lidar=True")
            # Camera path is unused under lidar_only forward.
            return [self.pp]
        raise ValueError(f"Unsupported probe decoder type: {decoder_type}")

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
        branch_mode=None,
    ):
        # LiDAR-only Full-L / Probe(L→L) / Probe(F→L): freeze-compatible lidar branch.
        if branch_mode == "lidar_only":
            if not self.lidar:
                raise ValueError("branch_mode=lidar_only requires a lidar-enabled HDMapNet")
            lidar_feature = self.pp(lidar_data, lidar_mask)
            return self.lidar_bevencode(lidar_feature)

        topdown = self.encode_camera_topdown(
            img, trans, rots, intrins, post_trans, post_rots, car_trans, yaw_pitch_roll
        )

        # Camera-only probe / camera model: never concatenate LiDAR.
        if branch_mode == "camera_only" or not self.lidar:
            return self.camera_bevencode(topdown)

        lidar_feature = self.pp(lidar_data, lidar_mask)
        fused = torch.cat([topdown, lidar_feature], dim=1)
        return self.bevencode(fused)
