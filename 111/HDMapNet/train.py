import os
import numpy as np
import sys
import logging
from time import time
from tensorboardX import SummaryWriter
import argparse

# Intermittent PyTorch 1.10 failure: NameError("name '_C' is not defined")
# during `import torch` under heavy multi-job GPU load. Clear partial import and retry.
def _import_torch():
    try:
        import torch as _torch
        return _torch
    except NameError:
        import sys as _sys
        for _k in list(_sys.modules):
            if _k == "torch" or _k.startswith("torch."):
                del _sys.modules[_k]
        import torch as _torch
        return _torch


torch = _import_torch()
from torch.optim.lr_scheduler import StepLR
from loss import SimpleLoss, DiscriminativeLoss

from data.dataset import semantic_dataset
from data.const import NUM_CLASSES
from evaluation.iou import get_batch_iou
from evaluation.angle_diff import calc_angle_diff
from model import get_model
from evaluate import onehot_encoding, eval_iou
from decoder_probe import (
    DECODER_PROBE_SOURCES,
    configure_decoder_probe_model,
    get_decoder_probe_trainable_parameters,
    normalize_legacy_camera_state_dict,
    reset_decoder_probe_parameters,
    resolve_decoder_probe_config,
    set_decoder_probe_train_mode,
    validate_decoder_probe_load_result,
)


def write_log(writer, ious, title, counter):
    writer.add_scalar(f'{title}/iou', torch.mean(ious[1:]), counter)

    for i, iou in enumerate(ious):
        writer.add_scalar(f'{title}/class_{i}/iou', iou, counter)


def load_model_checkpoint(model, checkpoint_path, strict=True):
    state_dict = {
        k.replace('module.', ''): v for k, v in torch.load(checkpoint_path, map_location='cpu').items()
    }
    state_dict, is_legacy_camera = normalize_legacy_camera_state_dict(
        state_dict, model.state_dict()
    )
    return model.load_state_dict(state_dict, strict=strict and not is_legacy_camera)


def validate_decoder_probe_args(args, probe_config):
    if args.branch_mode != probe_config.branch_mode:
        raise ValueError(
            f"Decoder probe source {args.probe_feature_source} requires "
            f"branch_mode={probe_config.branch_mode}, got {args.branch_mode}."
        )
    if not args.probe_encoder_checkpoint:
        raise ValueError("--probe_encoder_checkpoint is required when --train_decoder_only is set.")
    if args.finetune:
        raise ValueError("--train_decoder_only cannot be combined with --finetune.")
    if args.probe_feature_source == "fusion_base_camera" and args.model != "HDMapNet_fusion":
        raise ValueError(
            "fusion_base_camera probe requires --model HDMapNet_fusion "
            "(keep fusion structure, camera-only forward)."
        )
    if args.probe_feature_source == "camera_only_camera" and args.model not in (
        "HDMapNet_cam",
        "HDMapNet_fusion",
    ):
        raise ValueError(
            "camera_only_camera probe requires HDMapNet_cam or HDMapNet_fusion."
        )
    if args.probe_feature_source == "fusion_base_lidar" and args.model != "HDMapNet_fusion":
        raise ValueError(
            "fusion_base_lidar probe requires --model HDMapNet_fusion "
            "(keep fusion structure, lidar-only forward)."
        )
    if args.probe_feature_source == "lidar_only_lidar" and args.model not in (
        "HDMapNet_lidar",
        "HDMapNet_fusion",
    ):
        raise ValueError(
            "lidar_only_lidar probe requires HDMapNet_lidar or HDMapNet_fusion."
        )


def configure_decoder_probe(model, args):
    if not args.train_decoder_only:
        return None
    probe_config = resolve_decoder_probe_config(args.probe_feature_source)
    validate_decoder_probe_args(args, probe_config)
    # Always non-strict load; allowed-missing is checked by validate_*.
    load_result = load_model_checkpoint(
        model, args.probe_encoder_checkpoint, strict=False
    )
    validate_decoder_probe_load_result(
        probe_config, load_result.missing_keys, load_result.unexpected_keys
    )
    configure_decoder_probe_model(model, probe_config.decoder_type)
    reset_decoder_probe_parameters(model, probe_config.decoder_type)
    return probe_config


def train(args):
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(args.logdir, "results.log"),
                        filemode='w',
                        format='%(asctime)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    logging.getLogger('shapely.geos').setLevel(logging.CRITICAL)

    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler(sys.stdout))

    data_conf = {
        'num_channels': NUM_CLASSES + 1,
        'image_size': args.image_size,
        'xbound': args.xbound,
        'ybound': args.ybound,
        'zbound': args.zbound,
        'dbound': args.dbound,
        'thickness': args.thickness,
        'angle_class': args.angle_class,
    }

    train_loader, val_loader = semantic_dataset(args.version, args.dataroot, data_conf, args.bsz, args.nworkers)
    model = get_model(args.model, data_conf, args.instance_seg, args.embedding_dim, args.direction_pred, args.angle_class)

    probe_config = None
    if args.train_decoder_only:
        probe_config = configure_decoder_probe(model, args)
        logger.info(
            f"Decoder-only probe enabled: source={probe_config.feature_source}, "
            f"branch_mode={args.branch_mode}, model={args.model}"
        )
    elif args.finetune:
        model.load_state_dict(torch.load(args.modelf), strict=False)
        for name, param in model.named_parameters():
            if 'bevencode.up' in name or 'bevencode.layer3' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    model.cuda()

    if probe_config is not None:
        trainable = get_decoder_probe_trainable_parameters(model, probe_config.decoder_type)
        opt = torch.optim.Adam(trainable, lr=args.lr, weight_decay=args.weight_decay)
        logger.info(f"Trainable decoder params: {sum(p.numel() for p in trainable)}")
    else:
        opt = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    sched = StepLR(opt, 10, 0.1)
    writer = SummaryWriter(logdir=args.logdir)

    loss_fn = SimpleLoss(args.pos_weight).cuda()
    embedded_loss_fn = DiscriminativeLoss(args.embedding_dim, args.delta_v, args.delta_d).cuda()
    direction_loss_fn = torch.nn.BCELoss(reduction='none')

    if probe_config is not None:
        set_decoder_probe_train_mode(model, probe_config.decoder_type)
    else:
        model.train()
    counter = 0
    last_idx = len(train_loader) - 1
    for epoch in range(args.nepochs):
        for batchi, (imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans,
                     yaw_pitch_roll, semantic_gt, instance_gt, direction_gt) in enumerate(train_loader):
            t0 = time()
            opt.zero_grad()

            if probe_config is not None:
                set_decoder_probe_train_mode(model, probe_config.decoder_type)

            semantic, embedding, direction = model(
                imgs.cuda(),
                trans.cuda(),
                rots.cuda(),
                intrins.cuda(),
                post_trans.cuda(),
                post_rots.cuda(),
                lidar_data.cuda(),
                lidar_mask.cuda(),
                car_trans.cuda(),
                yaw_pitch_roll.cuda(),
                branch_mode=args.branch_mode if args.branch_mode != "fusion" else None,
            )

            semantic_gt = semantic_gt.cuda().float()
            instance_gt = instance_gt.cuda()
            seg_loss = loss_fn(semantic, semantic_gt)
            if args.instance_seg:
                var_loss, dist_loss, reg_loss = embedded_loss_fn(embedding, instance_gt)
            else:
                var_loss = 0
                dist_loss = 0
                reg_loss = 0

            if args.direction_pred:
                direction_gt = direction_gt.cuda()
                lane_mask = (1 - direction_gt[:, 0]).unsqueeze(1)
                direction_loss = direction_loss_fn(torch.softmax(direction, 1), direction_gt)
                direction_loss = (direction_loss * lane_mask).sum() / (lane_mask.sum() * direction_loss.shape[1] + 1e-6)
                angle_diff = calc_angle_diff(direction, direction_gt, args.angle_class)
            else:
                direction_loss = 0
                angle_diff = 0

            final_loss = seg_loss * args.scale_seg + var_loss * args.scale_var + dist_loss * args.scale_dist + direction_loss * args.scale_direction
            final_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            opt.step()
            counter += 1
            t1 = time()

            if counter % 10 == 0:
                intersects, union = get_batch_iou(onehot_encoding(semantic), semantic_gt)
                iou = intersects / (union + 1e-7)
                logger.info(f"TRAIN[{epoch:>3d}]: [{batchi:>4d}/{last_idx}]    "
                            f"Time: {t1-t0:>7.4f}    "
                            f"Loss: {final_loss.item():>7.4f}    "
                            f"IOU: {np.array2string(iou[1:].numpy(), precision=3, floatmode='fixed')}")

                write_log(writer, iou, 'train', counter)
                writer.add_scalar('train/step_time', t1 - t0, counter)
                writer.add_scalar('train/seg_loss', seg_loss, counter)
                writer.add_scalar('train/var_loss', var_loss, counter)
                writer.add_scalar('train/dist_loss', dist_loss, counter)
                writer.add_scalar('train/reg_loss', reg_loss, counter)
                writer.add_scalar('train/direction_loss', direction_loss, counter)
                writer.add_scalar('train/final_loss', final_loss, counter)
                writer.add_scalar('train/angle_diff', angle_diff, counter)

        iou = eval_iou(model, val_loader, branch_mode=args.branch_mode if args.branch_mode != "fusion" else None)
        logger.info(f"EVAL[{epoch:>2d}]:    "
                    f"IOU: {np.array2string(iou[1:].numpy(), precision=3, floatmode='fixed')}")

        write_log(writer, iou, 'eval', counter)
        model_name = os.path.join(args.logdir, f"model{epoch}.pt")
        torch.save(model.state_dict(), model_name)
        logger.info(f"{model_name} saved")
        if probe_config is not None:
            set_decoder_probe_train_mode(model, probe_config.decoder_type)
        else:
            model.train()

        sched.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HDMapNet training.')
    # logging config
    parser.add_argument("--logdir", type=str, default='./runs')

    # nuScenes config
    parser.add_argument('--dataroot', type=str, default='dataset/nuScenes/')
    parser.add_argument('--version', type=str, default='v1.0-mini', choices=['v1.0-trainval', 'v1.0-mini'])

    # model config
    parser.add_argument("--model", type=str, default='HDMapNet_cam')
    parser.add_argument(
        "--branch_mode",
        type=str,
        default="fusion",
        choices=["camera_only", "lidar_only", "fusion"],
        help="camera_only / lidar_only for single-branch Full-* and probes.",
    )

    # training config
    parser.add_argument("--nepochs", type=int, default=30)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--pos_weight", type=float, default=2.13)
    parser.add_argument("--bsz", type=int, default=4)
    parser.add_argument("--nworkers", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-7)

    # finetune config
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--modelf', type=str, default=None)

    # camera decoder probe (Satfor-style, adapted to camera+LiDAR)
    parser.add_argument('--train_decoder_only', action='store_true')
    parser.add_argument(
        '--probe_feature_source',
        type=str,
        default='',
        choices=[''] + list(DECODER_PROBE_SOURCES),
    )
    parser.add_argument('--probe_encoder_checkpoint', type=str, default='')

    # data config
    parser.add_argument("--thickness", type=int, default=5)
    parser.add_argument("--image_size", nargs=2, type=int, default=[128, 352])
    parser.add_argument("--xbound", nargs=3, type=float, default=[-30.0, 30.0, 0.15])
    parser.add_argument("--ybound", nargs=3, type=float, default=[-15.0, 15.0, 0.15])
    parser.add_argument("--zbound", nargs=3, type=float, default=[-10.0, 10.0, 20.0])
    parser.add_argument("--dbound", nargs=3, type=float, default=[4.0, 45.0, 1.0])

    # embedding config
    parser.add_argument('--instance_seg', action='store_true')
    parser.add_argument("--embedding_dim", type=int, default=16)
    parser.add_argument("--delta_v", type=float, default=0.5)
    parser.add_argument("--delta_d", type=float, default=3.0)

    # direction config
    parser.add_argument('--direction_pred', action='store_true')
    parser.add_argument('--angle_class', type=int, default=36)

    # loss config
    parser.add_argument("--scale_seg", type=float, default=1.0)
    parser.add_argument("--scale_var", type=float, default=1.0)
    parser.add_argument("--scale_dist", type=float, default=1.0)
    parser.add_argument("--scale_direction", type=float, default=0.2)

    args = parser.parse_args()
    train(args)
