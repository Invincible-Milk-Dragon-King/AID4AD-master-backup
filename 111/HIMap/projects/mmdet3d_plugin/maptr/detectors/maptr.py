import copy
from random import randint
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from mmcv.runner import force_fp32, auto_fp16
from mmcv.cnn import ConvModule

from einops import rearrange

from ..dense_heads.upsnetFPN import Upsample
from mmcv.utils import build_from_cfg
from mmdet.models import HEADS

from mmdet3d.ops import Voxelization, DynamicScatter
from mmdet3d.models import builder
@DETECTORS.register_module()
class MapTR(MVXTwoStageDetector):
    """MapTR.
    Args:
        video_test_mode (bool): Decide whether to use temporal information during inference.
    """

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 video_test_mode=False,
                 img_panoptic=None,  # following the name of VPS.
                 upsample=None,
                 other_config=None,
                 modality='vision',
                 # s m
                 cat_all_feats = False,
                 # s m
                 lidar_encoder=None,
                 force_camera_only=False,
                 force_lidar_only=False,
                 ):
        self.pts_bbox_head_config = pts_bbox_head
        self.pts_bbox_head_config['other_config'] = other_config if other_config is not None else dict()
        super(MapTR,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        # print("in MapTR, use_grid_mask: {}".format(use_grid_mask))  # True.
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False

        if img_panoptic is not None:
            self.panopticFPN = build_from_cfg(img_panoptic, HEADS)
            self.panopticFPN.init_weights()
            self.with_img_panoptic = True
        else:
            self.with_img_panoptic = False
        self.img_panoptic = img_panoptic
        if upsample is not None:
            self.upsample = Upsample(**upsample)
            self.upsample.init_weights()
        self.upsample_config = upsample
        self.other_config = other_config
        if self.other_config is None:
            self.other_config = dict()
            if train_cfg is not None:
                self.other_config = copy.deepcopy(train_cfg)

        if self.other_config.get("history_bev_manner", "rnn") in ['parallel_history', 'parallel_all']:
            if self.other_config.get("spatial_temporal_order", "sequential") == 'sequential':
                in_ch_mul = self.other_config.get("queue_length", 4)
                if self.other_config.get("history_bev_manner", "rnn") == 'parallel_history':
                    in_ch_mul = self.other_config.get("queue_length", 4) - 1
                self.tem_fuse_conv = ConvModule(
                    in_channels=self.pts_bbox_head_config.get("in_channels", 256) * in_ch_mul,
                    out_channels=self.pts_bbox_head_config.get("in_channels", 256),
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    conv_cfg=None,
                    act_cfg=None)
            if self.other_config.get("history_bev_manner", "rnn") == 'parallel_all':
                if self.other_config.get("grid_config", None) is not None:
                    self.create_grid_infos(**self.other_config.get("grid_config", None))
            self.grid = None

        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }
        self.modality = modality
        self.force_camera_only = force_camera_only
        self.force_lidar_only = force_lidar_only
        if self.force_camera_only and self.force_lidar_only:
            raise ValueError('force_camera_only and force_lidar_only cannot both be True')
        if self.modality in ('fusion','lidar') and lidar_encoder is not None :
            if isinstance(lidar_encoder["voxelize"]["voxel_size"][0], list):
                lidar_rand_ind = randint(0,2)
                lidar_encoder["voxelize"]["voxel_size"] = lidar_encoder["voxelize"]["voxel_size"][lidar_rand_ind]
                lidar_encoder["backbone"]["sparse_shape"] = lidar_encoder["backbone"]["sparse_shape"][lidar_rand_ind]
            if lidar_encoder["voxelize"].get("max_num_points", -1) > 0:
                voxelize_module = Voxelization(**lidar_encoder["voxelize"])
            else:
                voxelize_module = DynamicScatter(**lidar_encoder["voxelize"])
            self.lidar_modal_extractor = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": builder.build_middle_encoder(lidar_encoder["backbone"]),
                }
            )
            self.voxelize_reduce = lidar_encoder.get("voxelize_reduce", True)
            
        # s m
        self.cat_all_feats = cat_all_feats


    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:
            # print("extract_img_feat, img: {}".format(img.size()))
            # torch.Size([4, 6, 3, 480, 800])
            
            # input_shape = img.shape[-2:]
            # # update real input shape of each single img
            # for img_meta in img_metas:
            #     img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)  # apply some mask, is augmentation ???
                # print("extract_img_feat, after grid_mask, img: {}".format(img.size()))
                # torch.Size([24, 3, 480, 800])  # 24 = 4 * 6

            img_feats = self.img_backbone(img)
            # print("extract_img_feat, after img_backbone, img_feats: {}, "
            #       "len(img_feats): {}".format(type(img_feats), len(img_feats)))
            # len is 1, torch.Size([24, 2048, 15, 25])
            # for img_f in img_feats:
            #     print(img_f.size())
            # if output multi-scale feats, here batch size is 1.
            # torch.Size([6, 256, 120, 200])
            # torch.Size([6, 512, 60, 100])
            # torch.Size([6, 1024, 30, 50])
            # torch.Size([6, 2048, 15, 25])
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
            # print("after img_neck, img_feats: {}, len(img_feats): {}".format(type(img_feats), len(img_feats)))
            # # len is 1, torch.Size([24, 256, 15, 25])
            # for img_f in img_feats:
            #     print(img_f.size())
            # if output multi-scale feats, here batch size is 1.
            # torch.Size([6, 256, 120, 200])
            # torch.Size([6, 256, 60, 100])
            # torch.Size([6, 256, 30, 50])
            # torch.Size([6, 256, 15, 25])
            # torch.Size([6, 256, 8, 13])

        if self.with_img_panoptic:
            img_feats, img_feats_m = self.panopticFPN(img_feats[0:self.panopticFPN.num_levels])
            # print("after panopticFPN, img_feats: {}".format(img_feats.size()))
            # torch.Size([1, 6, 256, 120, 200])
            img_feats = (img_feats,)
        elif self.upsample_config is not None:
            if len(self.upsample.zoom_size) < len(img_feats):
                start_l = self.upsample.start_level
                end_l = start_l + len(self.upsample.zoom_size)
                img_feats = self.upsample(img_feats[start_l:end_l])
            else:  # default.
                img_feats = self.upsample(img_feats)
            # print("after upsample, img_feats: {}".format(img_feats.size()))
            # torch.Size([1, 6, 256, 120, 200])
            img_feats = (img_feats,)
        else:
            # defaultly utilize 4 scales features. may need to modify for other requirement.
            img_feats = img_feats[:4]

        img_feats_reshaped = []
        # print("img_feats_reshaped return size ......")
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
            # print(img_feats_reshaped[-1].size())
        # torch.Size([4, 6, 256, 15, 25])
        # if output multi-scale feats,
        # torch.Size([1, 6, 256, 120, 200])
        # torch.Size([1, 6, 256, 60, 100])
        # torch.Size([1, 6, 256, 30, 50])
        # torch.Size([1, 6, 256, 15, 25])
        # torch.Size([1, 6, 256, 8, 13])
        return img_feats_reshaped

    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images and points."""

        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        
        return img_feats


    def forward_pts_train(self,
                          pts_feats,
                          lidar_feat=None,
                          gt_bboxes_3d=None,
                          gt_labels_3d=None,
                          img_metas=None,
                          gt_bboxes_ignore=None,
                          prev_bev=None,
                          bev_embed=None,
                          gt_semantic_masks=None,
                          gt_instance_masks=None):
        """Forward function'
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
            prev_bev (torch.Tensor, optional): BEV features of previous frame.
        Returns:
            dict: Losses of each branch.
        """

        outs = self.pts_bbox_head(
            pts_feats, lidar_feat, img_metas, prev_bev, bev_embed=bev_embed)
        
        # s m
        if self.cat_all_feats:
            # print(len(gt_bboxes_3d))
            gt_bboxes_3d = gt_bboxes_3d+gt_bboxes_3d+gt_bboxes_3d
            gt_labels_3d = gt_labels_3d+gt_labels_3d+gt_labels_3d
            
            gt_semantic_masks = gt_semantic_masks.repeat(3,1,1,1)
            gt_instance_masks = gt_instance_masks + gt_instance_masks + gt_instance_masks
        
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses, _ = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas,
                                            gt_semantic_masks=gt_semantic_masks,
                                            gt_instance_masks=gt_instance_masks)
        if hasattr(self.pts_bbox_head, 'use_one2many_strategy') and self.pts_bbox_head.use_one2many_strategy:
            k_one2many = self.pts_bbox_head.k_one2many
            multi_gt_bboxes_3d = copy.deepcopy(gt_bboxes_3d)
            multi_gt_labels_3d = copy.deepcopy(gt_labels_3d)
            for i, (each_gt_bboxes_3d, each_gt_labels_3d) in enumerate(zip(multi_gt_bboxes_3d, multi_gt_labels_3d)):
                each_gt_bboxes_3d.instance_list = each_gt_bboxes_3d.instance_list * k_one2many
                each_gt_bboxes_3d.instance_labels = each_gt_bboxes_3d.instance_labels * k_one2many
                multi_gt_labels_3d[i] = each_gt_labels_3d.repeat(k_one2many)
            one2many_outs = outs['one2many_outs']
            one2many_outs['bev_sem_seg'] = None
            loss_one2many_inputs = [multi_gt_bboxes_3d, multi_gt_labels_3d, one2many_outs]
            multi_gt_instance_masks = copy.deepcopy(gt_instance_masks)
            multi_gt_instance_masks = [m.repeat(k_one2many, 1, 1) for m in multi_gt_instance_masks]
            loss_dict_one2many, _ = self.pts_bbox_head.loss(*loss_one2many_inputs, img_metas=img_metas,
                                                            gt_instance_masks=multi_gt_instance_masks)

            lambda_one2many = self.pts_bbox_head.lambda_one2many
            for key, value in loss_dict_one2many.items():
                if key + "_one2many" in losses.keys():
                    losses[key + "_one2many"] += value * lambda_one2many
                else:
                    losses[key + "_one2many"] = value * lambda_one2many
        return losses

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)
    
    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated.
           TODO. note that here only handles the history bev features of camera image, not include to Lidar
        """
        if not self.other_config.get("train_all_frames", False):
            # default.
            self.eval()

            with torch.no_grad():
                prev_bev = None
                bs, len_queue, num_cams, C, H, W = imgs_queue.shape
                # print("obtain_history_bev, imgs_queue: {}".format(imgs_queue.shape))
                imgs_queue = imgs_queue.reshape(bs*len_queue, num_cams, C, H, W)
                img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
                # here follows the BEV-Former, use the RNN-style to iteratively collect history BEV features.
                for i in range(len_queue):
                    img_metas = [each[i] for each in img_metas_list]
                    # print("i: {}, current img_metas: {}".format(i, img_metas))
                    if not img_metas[0]['prev_bev_exists']:
                        prev_bev = None
                    # img_feats = self.extract_feat(img=img, img_metas=img_metas)
                    img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                    # for img_f in img_feats:
                    #     print("obtain_history_bev, img_feats size(): {}".format(img_f.size()))
                    prev_bev = self.pts_bbox_head(
                        img_feats, img_metas=img_metas, prev_bev=prev_bev, only_bev=True)
                self.train()
                return prev_bev
        else:
            # TODO. currently have some backward issue, cannot utilize.
            with torch.autograd.set_detect_anomaly(True):
                prev_bev = None
                bs, len_queue, num_cams, C, H, W = imgs_queue.shape
                # print("obtain_history_bev, imgs_queue: {}".format(imgs_queue.shape))
                imgs_queue = imgs_queue.reshape(bs * len_queue, num_cams, C, H, W)
                img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
                # here follows the BEV-Former, use the RNN-style to iteratively collect history BEV features.
                for i in range(len_queue):
                    img_metas = [each[i] for each in img_metas_list]
                    # print("i: {}, current img_metas: {}".format(i, img_metas))
                    if not img_metas[0]['prev_bev_exists']:
                        prev_bev = None
                    # img_feats = self.extract_feat(img=img, img_metas=img_metas)
                    img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                    # for img_f in img_feats:
                    #     print("obtain_history_bev, img_feats size(): {}".format(img_f.size()))
                    if i == 0:
                        prev_bev_input = prev_bev
                    prev_bev = self.pts_bbox_head(
                        img_feats, img_metas=img_metas, prev_bev=prev_bev_input, only_bev=True)
                    prev_bev_input = prev_bev.detach()
                return prev_bev

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.lidar_modal_extractor["voxelize"](res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode="constant", value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
                    -1, 1
                )
                feats = feats.contiguous()

        return feats, coords, sizes
    @auto_fp16(apply_to=('points'), out_fp32=True)
    def extract_lidar_feat(self, points):
        feats, coords, sizes = self.voxelize(points)
        # voxel_features = self.lidar_modal_extractor["voxel_encoder"](feats, sizes, coords)
        batch_size = coords[-1, 0] + 1
        lidar_feat = self.lidar_modal_extractor["backbone"](feats, coords, batch_size, sizes=sizes)

        return lidar_feat

    def create_grid_infos(self, x, y, z, **kwargs):
        """Generate the grid information including the lower bound, interval,
        and size.

        Args:
            x (tuple(float)): Config of grid alone x axis in format of
                (lower_bound, upper_bound, interval).
            y (tuple(float)): Config of grid alone y axis in format of
                (lower_bound, upper_bound, interval).
            z (tuple(float)): Config of grid alone z axis in format of
                (lower_bound, upper_bound, interval).
            **kwargs: Container for other potential parameters
        """
        self.grid_lower_bound = torch.Tensor([cfg[0] for cfg in [x, y, z]])
        self.grid_interval = torch.Tensor([cfg[2] for cfg in [x, y, z]])
        # print("self.grid_lower_bound: {}".format(self.grid_lower_bound))
        # print("self.grid_interval: {}".format(self.grid_interval))
        # self.grid_size = torch.Tensor([(cfg[1] - cfg[0]) / cfg[2]
        #                                for cfg in [x, y, z]])

    def gen_grid(self, input, sensor2keyegos, bda=None, bda_adj=None):
        n, c, h, w = input.shape
        # _, v, _, _ = sensor2keyegos[0].shape
        # print("self.grid is None: {}".format(self.grid is None))  # None
        # print("gen_grid, input.shape: {}".format(input.shape))
        # if self.grid is not None:
            # print("gen_grid, self.grid: {}".format(self.grid.size()))
        # gen_grid, input.shape: torch.Size([2, 256, 120, 200])
        # gen_grid, self.grid: torch.Size([120, 200, 3])
        # TODO. note that here currently does not support multi-scale features.
        if self.grid is None:
            # generate grid
            xs = torch.linspace(
                0, w - 1, w, dtype=input.dtype,
                device=input.device).view(1, w).expand(h, w)
            ys = torch.linspace(
                0, h - 1, h, dtype=input.dtype,
                device=input.device).view(h, 1).expand(h, w)
            grid = torch.stack((xs, ys, torch.ones_like(xs)), -1)
            self.grid = grid
        else:
            grid = self.grid
        grid = grid.view(1, h, w, 3).expand(n, h, w, 3).view(n, h, w, 3, 1)
        # print("input: {}, grid: {}".format(input.size(), grid.size()))
        # print("sensor2keyegos 0: {}".format(sensor2keyegos[0].size()))
        # print("sensor2keyegos 1: {}".format(sensor2keyegos[1].size()))
        # print("bda: {}".format(bda.size()))
        # input: torch.Size([1, 80, 128, 128]), grid: torch.Size([1, 128, 128, 3, 1])
        # sensor2keyegos 0: torch.Size([1, 6, 4, 4])
        # sensor2keyegos 1: torch.Size([1, 6, 4, 4])
        # bda: torch.Size([1, 3, 3])

        # get transformation from current ego frame to adjacent ego frame
        # transformation from current camera frame to current ego frame
        c02l0 = sensor2keyegos[0][:, 0:1, :, :]

        # transformation from adjacent camera frame to current ego frame
        c12l0 = sensor2keyegos[1][:, 0:1, :, :]

        # add bev data augmentation
        if bda is not None:
            # TODO. curretly default we do not do the bev data augmentation.
            bda_ = torch.zeros((n, 1, 4, 4), dtype=grid.dtype).to(grid)
            bda_[:, :, :3, :3] = bda.unsqueeze(1)
            bda_[:, :, 3, 3] = 1
            c02l0 = bda_.matmul(c02l0)
            if bda_adj is not None:
                bda_ = torch.zeros((n, 1, 4, 4), dtype=grid.dtype).to(grid)
                bda_[:, :, :3, :3] = bda_adj.unsqueeze(1)
                bda_[:, :, 3, 3] = 1
            c12l0 = bda_.matmul(c12l0)

        # transformation from current ego frame to adjacent ego frame
        l02l1 = c02l0.matmul(torch.inverse(c12l0))[:, 0, :, :].view(
            n, 1, 1, 4, 4)
        '''
          c02l0 * inv(c12l0)
        = c02l0 * inv(l12l0 * c12l1)
        = c02l0 * inv(c12l1) * inv(l12l0)
        = l02l1 # c02l0==c12l1
        '''

        l02l1 = l02l1[:, :, :,
                      [True, True, False, True], :][:, :, :, :,
                                                    [True, True, False, True]]

        feat2bev = torch.zeros((3, 3), dtype=grid.dtype).to(grid)
        feat2bev[0, 0] = self.grid_interval[0]
        feat2bev[1, 1] = self.grid_interval[1]
        feat2bev[0, 2] = self.grid_lower_bound[0]
        feat2bev[1, 2] = self.grid_lower_bound[1]
        feat2bev[2, 2] = 1
        feat2bev = feat2bev.view(1, 3, 3)
        # print("feat2bev: {}, l02l1: {}".format(feat2bev, l02l1))
        tf = torch.inverse(feat2bev).matmul(l02l1.cuda()).matmul(feat2bev)

        # transform and normalize
        grid = tf.matmul(grid)
        normalize_factor = torch.tensor([w - 1.0, h - 1.0],
                                        dtype=input.dtype,
                                        device=input.device)
        grid = grid[:, :, :, :2, 0] / normalize_factor.view(1, 1, 1,
                                                            2) * 2.0 - 1.0
        return grid

    @force_fp32()
    def shift_feature(self, input, sensor2keyegos, bda=None, bda_adj=None):
        grid = self.gen_grid(input, sensor2keyegos, bda, bda_adj=bda_adj)
        # print("shift_feature, grid: {}".format(grid.size()))
        # shift_feature, grid: torch.Size([1, 128, 128, 2])
        output = F.grid_sample(input, grid.to(input.dtype), align_corners=True)
        return output

    def prepare_shift_feature(self, img_metas, len_queue):
        # generate sensor2keyegos
        global2keyego_list = []
        ego2global_list = []
        sensor2ego_list = []
        for bi in range(len(img_metas)):
            # the last 0 is for the first view.
            # 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
            view_index = self.other_config.get("keyego_view_index", 0)
            keyego2global = img_metas[bi][len_queue - 1]['ego2global'][view_index] \
                .unsqueeze(0).unsqueeze(0).unsqueeze(0)
            # 1, 1, 1, 4, 4
            global2keyego = torch.inverse(keyego2global.double())
            global2keyego_list.append(global2keyego)
            b_ego2global_list = []
            b_sensor2ego_list = []
            for li in range(0, len_queue):
                curr_ego2global = [im[None] for im in img_metas[bi][li]['ego2global']]
                curr_ego2global = torch.cat(curr_ego2global, dim=0)  # should be 6, x, x
                b_ego2global_list.append(curr_ego2global[None])
                curr_sensor2ego = [im[None] for im in img_metas[bi][li]['sensor2ego']]
                curr_sensor2ego = torch.cat(curr_sensor2ego, dim=0)
                b_sensor2ego_list.append(curr_sensor2ego[None])
            b_ego2global_list = torch.cat(b_ego2global_list, dim=0)  # should be queue-1, 6, x, x
            ego2global_list.append(b_ego2global_list[None])
            b_sensor2ego_list = torch.cat(b_sensor2ego_list, dim=0)
            sensor2ego_list.append(b_sensor2ego_list[None])
        global2keyego_list = torch.cat(global2keyego_list, dim=0)
        ego2global_list = torch.cat(ego2global_list, dim=0)
        sensor2ego_list = torch.cat(sensor2ego_list, dim=0)
        # print("global2keyego_list size: {}".format(global2keyego_list.size()))  # B, 1, 1, 4, 4
        # print("ego2global_list size: {}".format(ego2global_list.size()))
        # print("sensor2ego_list size: {}".format(sensor2ego_list.size()))
        # img_feats_list[scale_i] size should be int(B/len_queue), len_queue, int(BN / B), C, H, W)
        # global2keyego_list size: torch.Size([2, 1, 1, 4, 4])
        # ego2global_list size: torch.Size([2, 2, 6, 4, 4])
        # sensor2ego_list size: torch.Size([2, 2, 6, 4, 4])
        sensor2keyegos = global2keyego_list @ ego2global_list.double() @ sensor2ego_list.double()
        sensor2keyegos = sensor2keyegos.float()
        # print("sensor2keyegos: {}".format(sensor2keyegos.size()))
        # sensor2keyegos: torch.Size([2, 2, 6, 4, 4])
        return sensor2keyegos

    # @auto_fp16(apply_to=('img', 'points'))
    @force_fp32(apply_to=('img','points','prev_bev'))
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None,
                      gt_semantic_masks=None,
                      gt_instance_masks=None,
                      gt_instance_ids=None,
                      uni_instances=None,
                      ):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        lidar_feat = None
        if (self.modality == 'fusion' or self.modality == 'lidar') and not self.force_camera_only:
            lidar_feat = self.extract_lidar_feat(points)

        # print("forward_train, input img: {}".format(img.size()))
        # torch.Size([4, 1, 6, 3, 480, 800])
        # torch.Size([4, 4, 6, 3, 480, 800]) for len_queue = 4
        len_queue = img.size(1)
        # print("forward_train, len_queue: {}".format(len_queue))  # 1
        # prev_img = img[:, :-1, ...]
        # img = img[:, -1, ...]
        # print("forward_train, prev_img: {}".format(prev_img.size()))
        # torch.Size([4, 0, 6, 3, 480, 800])
        # print("forward_train, img: {}".format(img.size()))
        # torch.Size([4, 6, 3, 480, 800])
        # if gt_semantic_masks is not None:
        #     print("gt_semantic_mask: {}".format(gt_semantic_masks.size()))
            # gt_semantic_mask: torch.Size([2, 1, 3, 200, 100])   # 2 is batch size, 1 is queue length
            # gt_semantic_masks: torch.Size([1, 4, 3, 200, 100])
        # TODO. here currently only support forwaring single-frame into decoder, otherwise need to change code.
        gsm = gt_semantic_masks[:, -1, :, :, :].contiguous() if gt_semantic_masks is not None else None
        # if gt_instance_masks is not None:
        #     # len(gt_instance_masks) is the len_queue.
        #     print("gt_instance_mask: {}".format(len(gt_instance_masks[0])))
        #     for gi in range(len(gt_instance_masks[0])):
        #         print(gt_instance_masks[0][gi].size())
        #     # gt_instance_mask: 2
        #     # torch.Size([5, 200, 100]); torch.Size([3, 200, 100])
        gim = gt_instance_masks[0] if gt_instance_masks is not None else None
        # print('gt_labels_3d: {}, {}'.format(len(gt_labels_3d), gt_labels_3d))
        # gt_labels_3d: 2, [tensor([0, 0, 0, 2, 2], device='cuda:0'), tensor([0, 2, 2], device='cuda:0')]

        if self.other_config.get("history_bev_manner", "rnn") == 'rnn':
            # default.
            prev_img = img[:, :-1, ...]
            img = img[:, -1, ...]

            prev_img_metas = copy.deepcopy(img_metas)
            # print("img_metas: {}".format(img_metas))
            # print("len(img_metas): {}, img_metas[0].keys(): {}".format(len(img_metas), img_metas[0][0].keys()))
            # len(img_metas): 2, img_metas[0].keys(): dict_keys(['filename', 'ori_shape', 'img_shape', 'lidar2img',
            # 'pad_shape', 'scale_factor', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'sample_idx', 'prev_idx',
            # 'next_idx', 'pts_filename', 'scene_token', 'can_bus', 'lidar2global', 'camera2ego', 'camera_intrinsics',
            # 'img_aug_matrix', 'lidar2ego', 'prev_bev', 'prev_bev_exists'])
            # each frame has lidar2img, lidar2global, camera2ego, camera_intrinsics, lidar2ego
            # prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)
            # import pdb;pdb.set_trace()
            prev_bev = self.obtain_history_bev(prev_img, prev_img_metas) if len_queue>1 else None
            # obtain_history_bev includes the process of extract_feat and pts_bbox_head
            # if len_queue > 1:
            #     print("prev_bev: {}".format(prev_bev.size()))

            img_metas = [each[len_queue-1] for each in img_metas]
            # print("img_metas[0]['prev_bev_exists']: {}".format(img_metas[0]['prev_bev_exists']))
            if not img_metas[0]['prev_bev_exists']:
                prev_bev = None
                
            img_feats = None
            if self.modality != 'lidar' and not self.force_lidar_only:
                img_feats = self.extract_feat(img=img, img_metas=img_metas)
            losses = dict()
            losses_pts = self.forward_pts_train(img_feats, lidar_feat, gt_bboxes_3d,
                                                gt_labels_3d, img_metas,
                                                gt_bboxes_ignore, prev_bev, gt_semantic_masks=gsm,
                                                gt_instance_masks=gim)
        elif self.other_config.get("history_bev_manner", "rnn") == 'parallel_history':
            bs, len_queue, num_cams, C, H, W = img.shape
            imgs_queue = img.reshape(bs * len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
            # print("img_feats_list: {}, {}".format(len(img_feats_list), img_feats_list[0].size()))
            # prev_bev = None
            # bs, len_queue, num_cams, C, H, W = prev_img.shape
            # # print("obtain_history_bev, imgs_queue: {}".format(imgs_queue.shape))
            # prev_imgs_queue = prev_img.reshape(bs * len_queue, num_cams, C, H, W)
            # img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
            # here follows the BEV-Former, use the RNN-style to iteratively collect history BEV features.
            prev_bev_list = []
            img_metas_list = copy.deepcopy(img_metas)
            for i in range(len_queue - 1):
                curr_img_metas = [each[i] for each in img_metas_list]
                # print("i: {}, current img_metas: {}".format(i, img_metas))
                # if not img_metas[0]['prev_bev_exists']:
                #     prev_bev = None
                # img_feats = self.extract_feat(img=img, img_metas=img_metas)
                curr_img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                # for img_f in img_feats:
                #     print("obtain_history_bev, img_feats size(): {}".format(img_f.size()))
                curr_bev = self.pts_bbox_head(
                    curr_img_feats, img_metas=curr_img_metas, prev_bev=None, only_bev=True)
                prev_bev_list.append(curr_bev)
            prev_bev_list = torch.cat(prev_bev_list, dim=-1)
            # print("prev_bev_list: {}".format(prev_bev_list.size()))
            prev_bev_list = rearrange(prev_bev_list, "b (h w) c -> b c h w",
                                      h=self.pts_bbox_head_config.get("bev_h"),
                                      w=self.pts_bbox_head_config.get("bev_w"))
            # print("reshape prev_bev_list: {}".format(prev_bev_list.size()))
            prev_bev = self.tem_fuse_conv(prev_bev_list)
            prev_bev = rearrange(prev_bev, "b c h w -> b (h w) c")
            # print("prev_bev: {}".format(prev_bev.size()))

            img_metas = [each[len_queue - 1] for each in img_metas_list]
            # print("img_metas[0]['prev_bev_exists']: {}".format(img_metas[0]['prev_bev_exists']))
            if not img_metas[0]['prev_bev_exists']:
                prev_bev = None
            img_feats = [each[:, len_queue - 1] for each in img_feats_list]
            bev = self.pts_bbox_head(
                img_feats, img_metas=img_metas, prev_bev=prev_bev, only_bev=True
            )
            losses = dict()
            losses_pts = self.forward_pts_train(pts_feats=img_feats, lidar_feat=lidar_feat, gt_bboxes_3d=gt_bboxes_3d,
                                                gt_labels_3d=gt_labels_3d, img_metas=img_metas,
                                                gt_bboxes_ignore=gt_bboxes_ignore, prev_bev=None, bev_embed=bev,
                                                gt_semantic_masks=gsm, gt_instance_masks=gim)
        elif self.other_config.get("history_bev_manner", "rnn") == 'parallel_all':
            bs, len_queue, num_cams, C, H, W = img.shape
            imgs_queue = img.reshape(bs * len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
            # print("img_feats_list: {}, {}".format(len(img_feats_list), img_feats_list[0].size()))
            # prev_bev = None
            # bs, len_queue, num_cams, C, H, W = prev_img.shape
            # # print("obtain_history_bev, imgs_queue: {}".format(imgs_queue.shape))
            # prev_imgs_queue = prev_img.reshape(bs * len_queue, num_cams, C, H, W)
            # img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
            bev_list = []
            img_metas_list = copy.deepcopy(img_metas)
            if self.other_config.get("align_view_transformation_option", 'None') == 'before':
                # print("len(img_metas): {}, img_metas[0].keys(): {}".format(len(img_metas), img_metas[0][0].keys()))
                # print(img_metas[0][0]['ego2global'])
                # [tensor([[-1.6996e-01, -9.8541e-01, -9.1065e-03,  3.4975e+02],
                #         [ 9.8515e-01, -1.7013e-01,  2.3303e-02,  1.8214e+03],
                #         [-2.4512e-02, -5.0107e-03,  9.9969e-01,  0.0000e+00],
                #         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]]), tensor([[-1.6974e-01, -9.8544e-01, -9.3439e-03,  3.4974e+02],
                #         [ 9.8518e-01, -1.6992e-01,  2.3418e-02,  1.8215e+03],
                #         [-2.4665e-02, -5.2303e-03,  9.9968e-01,  0.0000e+00],
                #         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]]), tensor([[-1.7019e-01, -9.8537e-01, -8.8776e-03,  3.4976e+02],
                #         [ 9.8511e-01, -1.7035e-01,  2.3192e-02,  1.8214e+03],
                #         [-2.4365e-02, -4.7984e-03,  9.9969e-01,  0.0000e+00],
                #         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]]), tensor([[-1.6925e-01, -9.8552e-01, -9.9057e-03,  3.4972e+02],
                #         [ 9.8526e-01, -1.6945e-01,  2.3622e-02,  1.8216e+03],
                #         [-2.4958e-02, -5.7616e-03,  9.9967e-01,  0.0000e+00],
                #         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]]), tensor([[-1.6894e-01, -9.8557e-01, -1.0243e-02,  3.4971e+02],
                #         [ 9.8531e-01, -1.6915e-01,  2.3732e-02,  1.8217e+03],
                #         [-2.5122e-02, -6.0831e-03,  9.9967e-01,  0.0000e+00],
                #         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]]), tensor([[-1.6954e-01, -9.8548e-01, -9.5686e-03,  3.4973e+02],
                #         [ 9.8521e-01, -1.6972e-01,  2.3508e-02,  1.8215e+03],
                #         [-2.4791e-02, -5.4415e-03,  9.9968e-01,  0.0000e+00],
                #         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])]
                # generate sensor2keyegos
                sensor2keyegos = self.prepare_shift_feature(img_metas, len_queue)
                view_num = img_feats_list[0].size(2)
                for scale_i in range(len(img_feats_list)):
                    for adj_id in range(0, len_queue - 1):
                        for v_i in range(view_num):
                            shifted_feature = \
                                self.shift_feature(img_feats_list[scale_i][:, adj_id, v_i].clone(),
                                                   [sensor2keyegos[:, -1, v_i].unsqueeze(1),
                                                    sensor2keyegos[:, adj_id, v_i].unsqueeze(1)],
                                                   bda=None)
                            img_feats_list[scale_i][:, adj_id, v_i] = shifted_feature
            if self.other_config.get("spatial_temporal_order", "sequential") == 'sequential':
                # defualt, first get all bev_list, then use e.g. conv to fuse into single frame.
                # spatial-spatial-spatial-temporal-temporal-temporal
                for i in range(len_queue):
                    curr_img_metas = [each[i] for each in img_metas_list]
                    # print("i: {}, current img_metas: {}".format(i, img_metas))
                    # if not img_metas[0]['prev_bev_exists']:
                    #     prev_bev = None
                    # img_feats = self.extract_feat(img=img, img_metas=img_metas)
                    curr_img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                    # for img_f in curr_img_feats:
                    #     print("obtain_history_bev, img_feats size(): {}".format(img_f.size()))
                    curr_bev = self.pts_bbox_head(
                        curr_img_feats, img_metas=curr_img_metas, prev_bev=None, only_bev=True)
                    curr_bev = rearrange(curr_bev, "b (h w) c -> b c h w",
                                         h=self.pts_bbox_head_config.get("bev_h"),
                                         w=self.pts_bbox_head_config.get("bev_w"))
                    bev_list.append(curr_bev)
                if self.other_config.get("align_view_transformation_option", 'None') == 'after':
                    # print("len(img_metas): {}, img_metas[0].keys(): {}".format(len(img_metas), img_metas[0][0].keys()))
                    # print(img_metas[0][0]['ego2global'])
                    # generate sensor2keyegos
                    sensor2keyegos = self.prepare_shift_feature(img_metas, len_queue)
                    bkvi = self.other_config.get("bev_keyego_view_index", 0)
                    for adj_id in range(0, len_queue - 1):
                        shifted_bev = self.shift_feature(bev_list[adj_id],
                                                         [sensor2keyegos[:, -1, bkvi].unsqueeze(1),
                                                          sensor2keyegos[:, adj_id, bkvi].unsqueeze(1)],
                                                         bda=None)
                        bev_list[adj_id] = shifted_bev
                bev_list = torch.cat(bev_list, dim=1)
                # print("bev_list: {}".format(bev_list.size()))
                # bev_list = rearrange(bev_list, "b (h w) c -> b c h w",
                #                      h=self.pts_bbox_head_config.get("bev_h"),
                #                      w=self.pts_bbox_head_config.get("bev_w"))
                # print("reshape prev_bev_list: {}".format(bev_list.size()))
                bev = self.tem_fuse_conv(bev_list)
                bev = rearrange(bev, "b c h w -> b (h w) c")
                # print("bev: {}".format(bev.size()))
            elif self.other_config.get("spatial_temporal_order", "sequential") == 'iterative':
                # spatial-temporal-spatial-temporal...
                bev = self.pts_bbox_head(
                    img_feats_list, img_metas=img_metas_list, prev_bev=None, only_bev=True,
                    bev_input_multi_frames=True)
            else:
                assert 1 == 2, "INVALID spatial_temporal_order !!!!!!"

            img_metas = [each[len_queue - 1] for each in img_metas_list]
            # print("img_metas[0]['prev_bev_exists']: {}".format(img_metas[0]['prev_bev_exists']))
            if not img_metas[0]['prev_bev_exists']:
                prev_bev = None
            img_feats = [each[:, len_queue - 1] for each in img_feats_list]
            losses = dict()
            losses_pts = self.forward_pts_train(pts_feats=img_feats, lidar_feat=lidar_feat, gt_bboxes_3d=gt_bboxes_3d,
                                                gt_labels_3d=gt_labels_3d, img_metas=img_metas,
                                                gt_bboxes_ignore=gt_bboxes_ignore, prev_bev=None, bev_embed=bev,
                                                gt_semantic_masks=gsm, gt_instance_masks=gim)
        else:
            assert 1 == 2, "INVALID history_bev_manner !!!!!!"

        losses.update(losses_pts)
        return losses

    def forward_test(self, img_metas, img=None, points=None,  **kwargs):
        # print("forward_test, img: {}".format(len(img)))  # 1.
        # print("forward_test, img_metas: {}".format(len(img_metas)))  # 1.
        # print("kwargs, {}".format(kwargs.keys()))
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img
        points = [points] if points is None else points
        if img_metas[0][0]['scene_token'] != self.prev_frame_info['scene_token']:
            # the first sample of each scene is truncated
            self.prev_frame_info['prev_bev'] = None
        # update idx
        self.prev_frame_info['scene_token'] = img_metas[0][0]['scene_token']

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None

        if not self.other_config.get("test_load_prev_img", False):
            # default.
            # Get the delta of ego position and angle between two timestamps.
            tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])
            tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][-1])
            if self.prev_frame_info['prev_bev'] is not None:
                img_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
                img_metas[0][0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
            else:
                img_metas[0][0]['can_bus'][-1] = 0
                img_metas[0][0]['can_bus'][:3] = 0

            new_prev_bev, bbox_results = self.simple_test(
                img_metas[0], img[0], points[0], prev_bev=self.prev_frame_info['prev_bev'], **kwargs)
            # During inference, we save the BEV features and ego motion of each timestamp.
            self.prev_frame_info['prev_pos'] = tmp_pos
            self.prev_frame_info['prev_angle'] = tmp_angle
            self.prev_frame_info['prev_bev'] = new_prev_bev
            return bbox_results
        else:
            """Test function without augmentaiton."""
            lidar_feat = None
            if (self.modality == 'fusion' or self.modality == 'lidar') and not self.force_camera_only:
                lidar_feat = self.extract_lidar_feat(points)

            if self.other_config.get("history_bev_manner", "rnn") == 'rnn':
                img_feats = self.extract_feat(img=img, img_metas=img_metas)

                bbox_final_list = [dict() for i in range(len(img_metas))]
                outs = self.pts_bbox_head(img_feats, lidar_feat, img_metas, prev_bev=self.prev_frame_info['prev_bev'])
                bbox_list = self.pts_bbox_head.get_bboxes(
                    outs, img_metas, rescale=kwargs['rescale'])
                bbox_results = [
                    self.pred2result(bboxes, scores, labels, pts)
                    for bboxes, scores, labels, pts in bbox_list
                ]
                new_prev_bev = outs['bev_embed']
                for result_dict, pts_bbox in zip(bbox_final_list, bbox_results):
                    result_dict['pts_bbox'] = pts_bbox
                # self.prev_frame_info['prev_pos'] = tmp_pos
                # self.prev_frame_info['prev_angle'] = tmp_angle
                self.prev_frame_info['prev_bev'] = new_prev_bev
                return bbox_final_list
            elif self.other_config.get("history_bev_manner", "rnn") == 'parallel_all':
                bs, len_queue, num_cams, C, H, W = img.shape
                # print("img.shape: {}".format(img.shape)) # torch.Size([1, 2, 6, 3, 480, 800])
                imgs_queue = img.reshape(bs * len_queue, num_cams, C, H, W)
                img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
                # print("img_feats_list: {}, {}".format(len(img_feats_list), img_feats_list[0].size()))
                # prev_bev = None
                # bs, len_queue, num_cams, C, H, W = prev_img.shape
                # # print("obtain_history_bev, imgs_queue: {}".format(imgs_queue.shape))
                # prev_imgs_queue = prev_img.reshape(bs * len_queue, num_cams, C, H, W)
                # img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
                bev_list = []
                img_metas_list = copy.deepcopy(img_metas)
                if self.other_config.get("align_view_transformation_option", 'None') == 'before':
                    # print("len(img_metas): {}, img_metas[0].keys(): {}".format(len(img_metas), img_metas[0][0].keys()))
                    # print(img_metas[0][0]['ego2global'])
                    # generate sensor2keyegos
                    sensor2keyegos = self.prepare_shift_feature(img_metas, len_queue)
                    view_num = img_feats_list[0].size(2)
                    for scale_i in range(len(img_feats_list)):
                        for adj_id in range(0, len_queue - 1):
                            for v_i in range(view_num):
                                shifted_feature = \
                                    self.shift_feature(img_feats_list[scale_i][:, adj_id, v_i].clone(),
                                                       [sensor2keyegos[:, -1, v_i].unsqueeze(1),
                                                        sensor2keyegos[:, adj_id, v_i].unsqueeze(1)],
                                                       bda=None)
                                img_feats_list[scale_i][:, adj_id, v_i] = shifted_feature
                if self.other_config.get("spatial_temporal_order", "sequential") == 'sequential':
                    # defualt, first get all bev_list, then use e.g. conv to fuse into single frame.
                    # spatial-spatial-spatial-temporal-temporal-temporal
                    for i in range(len_queue):
                        curr_img_metas = [each[i] for each in img_metas_list]
                        # print("i: {}, current img_metas: {}".format(i, img_metas))
                        # if not img_metas[0]['prev_bev_exists']:
                        #     prev_bev = None
                        # img_feats = self.extract_feat(img=img, img_metas=img_metas)
                        curr_img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                        # for img_f in curr_img_feats:
                        #     print("obtain_history_bev, img_feats size(): {}".format(img_f.size()))
                        curr_bev = self.pts_bbox_head(
                            curr_img_feats, img_metas=curr_img_metas, prev_bev=None, only_bev=True)
                        curr_bev = rearrange(curr_bev, "b (h w) c -> b c h w",
                                             h=self.pts_bbox_head_config.get("bev_h"),
                                             w=self.pts_bbox_head_config.get("bev_w"))
                        bev_list.append(curr_bev)
                    if self.other_config.get("align_view_transformation_option", 'None') == 'after':
                        # print("len(img_metas): {}, img_metas[0].keys(): {}".format(len(img_metas), img_metas[0][0].keys()))
                        # print(img_metas[0][0]['ego2global'])
                        # generate sensor2keyegos
                        sensor2keyegos = self.prepare_shift_feature(img_metas, len_queue)
                        bkvi = self.other_config.get("bev_keyego_view_index", 0)
                        for adj_id in range(0, len_queue - 1):
                            shifted_bev = self.shift_feature(bev_list[adj_id],
                                                             [sensor2keyegos[:, -1, bkvi].unsqueeze(1),
                                                              sensor2keyegos[:, adj_id, bkvi].unsqueeze(1)],
                                                             bda=None)
                            bev_list[adj_id] = shifted_bev
                    bev_list = torch.cat(bev_list, dim=1)
                    # print("bev_list: {}".format(bev_list.size()))
                    # bev_list = rearrange(bev_list, "b (h w) c -> b c h w",
                    #                      h=self.pts_bbox_head_config.get("bev_h"),
                    #                      w=self.pts_bbox_head_config.get("bev_w"))
                    # print("reshape prev_bev_list: {}".format(bev_list.size()))
                    bev = self.tem_fuse_conv(bev_list)
                    bev = rearrange(bev, "b c h w -> b (h w) c")
                    # print("bev: {}".format(bev.size()))
                elif self.other_config.get("spatial_temporal_order", "sequential") == 'iterative':
                    # spatial-temporal-spatial-temporal...
                    bev = self.pts_bbox_head(
                        img_feats_list, img_metas=img_metas_list, prev_bev=None, only_bev=True,
                        bev_input_multi_frames=True)

                bbox_final_list = [dict() for i in range(len(img_metas_list))]
                img_metas = [each[len_queue - 1] for each in img_metas_list]
                # print("img_metas[0]['prev_bev_exists']: {}".format(img_metas[0]['prev_bev_exists']))
                if not img_metas[0]['prev_bev_exists']:
                    prev_bev = None
                img_feats = [each[:, len_queue - 1] for each in img_feats_list]
                # TODO. note that here the lidar feat does not involve, need upgrade code for utilizing lidar feature
                #  when using the temporal camera features.
                outs = self.pts_bbox_head(
                    mlvl_feats=img_feats, lidar_feat=lidar_feat, img_metas=img_metas, prev_bev=None, bev_embed=bev)
                bbox_list = self.pts_bbox_head.get_bboxes(
                    outs, img_metas, rescale=kwargs['rescale'])
                bbox_results = [
                    self.pred2result(bboxes, scores, labels, pts)
                    for bboxes, scores, labels, pts in bbox_list
                ]
                # new_prev_bev = outs['bev_embed']
                for result_dict, pts_bbox in zip(bbox_final_list, bbox_results):
                    result_dict['pts_bbox'] = pts_bbox
                # self.prev_frame_info['prev_pos'] = tmp_pos
                # self.prev_frame_info['prev_angle'] = tmp_angle
                # self.prev_frame_info['prev_bev'] = new_prev_bev
                return bbox_final_list


    def pred2result(self, bboxes, scores, labels, pts, pseudo_pts=None, attrs=None, masks=None):
        """Convert detection results to a list of numpy arrays.

        Args:
            bboxes (torch.Tensor): Bounding boxes with shape of (n, 5).
            labels (torch.Tensor): Labels with shape of (n, ).
            scores (torch.Tensor): Scores with shape of (n, ).
            attrs (torch.Tensor, optional): Attributes with shape of (n, ). \
                Defaults to None.

        Returns:
            dict[str, torch.Tensor]: Bounding box results in cpu mode.

                - boxes_3d (torch.Tensor): 3D boxes.
                - scores (torch.Tensor): Prediction scores.
                - labels_3d (torch.Tensor): Box labels.
                - attrs_3d (torch.Tensor, optional): Box attributes.
        """
        result_dict = dict(
            boxes_3d=bboxes.to('cpu'),
            scores_3d=scores.cpu(),
            labels_3d=labels.cpu(),
            pts_3d=pts.to('cpu'))

        if pseudo_pts is not None:
            result_dict['pseudo_pts_3d'] = pseudo_pts
        if attrs is not None:
            result_dict['attrs_3d'] = attrs.cpu()

        if masks is not None:
            if masks.size(0) == scores.size(0):
                result_dict['masks_3d'] = masks.to('cpu')
            else:
                result_dict['pred_sem_map'] = masks.to('cpu')

        return result_dict
    def simple_test_pts(self, x, lidar_feat=None, img_metas=None, prev_bev=None, rescale=False):
        """Test function"""
        if self.other_config.get("history_bev_manner", "rnn") == 'rnn':
            outs = self.pts_bbox_head(x, lidar_feat, img_metas, prev_bev=prev_bev)
        elif self.other_config.get("history_bev_manner", "rnn") == 'parallel_all':
            # TODO. note that here only support queue_length = 2
            if prev_bev is not None:
                curr_bev = self.pts_bbox_head(x, lidar_feat, img_metas, prev_bev=None, only_bev=True)
                bev_list = torch.cat([prev_bev.permute(1, 0, 2), curr_bev], dim=-1)
                # print("bev_list: {}".format(bev_list.size()))
                bev_list = rearrange(bev_list, "b (h w) c -> b c h w",
                                     h=self.pts_bbox_head_config.get("bev_h"),
                                     w=self.pts_bbox_head_config.get("bev_w"))
                # print("reshape prev_bev_list: {}".format(bev_list.size()))
                bev = self.tem_fuse_conv(bev_list)
                bev = rearrange(bev, "b c h w -> b (h w) c")
            else:
                # TODO. or we can concat the curr_bev with itself.
                bev = self.pts_bbox_head(x, lidar_feat, img_metas, prev_bev=None, only_bev=True)
            outs = self.pts_bbox_head(x, lidar_feat, img_metas, prev_bev=None, bev_embed=bev)
        else:
            assert 1 == 2, "INVALID history_bev_manner in inference stage !!!!!!"

        # print("simple_test_pts, outs keys: {}".format(outs.keys()))
        # outs keys: dict_keys(['bev_embed', 'all_cls_scores', 'all_bbox_preds',
        # 'all_pts_preds', 'enc_cls_scores', 'enc_bbox_preds', 'enc_pts_preds'])

        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        # print("simple_test_pts, len(bbox_list): {}".format(len(bbox_list)))
        # [bboxes, scores, labels, pts]
        # print(bbox_list[0][0].size(), bbox_list[0][1].size(), bbox_list[0][2].size(),
        #       bbox_list[0][3].size(), bbox_list[0][4].size())
        # torch.Size([50, 4]) torch.Size([50]) torch.Size([50]) torch.Size([50, 20, 2])
        # print(bbox_list[0][3][0])
        # print(bbox_list[0][3][1])
        # print(bbox_list[0][3][20])
        # assert 1 == 2
        bbox_results = [
            self.pred2result(bboxes, scores, labels, pts, pseudo_pts=pseudo_pts, masks=masks)
            for bboxes, scores, labels, pts, pseudo_pts, masks in bbox_list
        ]
        # import pdb;pdb.set_trace()
        return outs['bev_embed'], bbox_results
    def simple_test(self, img_metas, img=None, points=None, prev_bev=None, rescale=False, **kwargs):
        """Test function without augmentaiton."""
        lidar_feat = None
        if (self.modality =='fusion' or self.modality =='lidar') and not self.force_camera_only:
            lidar_feat = self.extract_lidar_feat(points)
        img_feats = None
        if self.modality != 'lidar' and not self.force_lidar_only:
            img_feats = self.extract_feat(img=img, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, bbox_pts = self.simple_test_pts(
            img_feats, lidar_feat, img_metas, prev_bev, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return new_prev_bev, bbox_list


@DETECTORS.register_module()
class MapTR_fp16(MapTR):
    """
    The default version BEVFormer currently can not support FP16. 
    We provide this version to resolve this issue.
    """
    # @auto_fp16(apply_to=('img', 'prev_bev', 'points'))
    @force_fp32(apply_to=('img','points','prev_bev'))
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None,
                      prev_bev=None,
                      ):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        # import pdb;pdb.set_trace()
        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore, prev_bev=prev_bev)
        losses.update(losses_pts)
        return losses


    def val_step(self, data, optimizer):
        """
        In BEVFormer_fp16, we use this `val_step` function to inference the `prev_pev`.
        This is not the standard function of `val_step`.
        """

        img = data['img']
        img_metas = data['img_metas']
        img_feats = self.extract_feat(img=img,  img_metas=img_metas)
        prev_bev = data.get('prev_bev', None)
        prev_bev = self.pts_bbox_head(img_feats, img_metas, prev_bev=prev_bev, only_bev=True)
        return prev_bev
