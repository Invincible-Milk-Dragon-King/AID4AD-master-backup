import json
import os

import torch
import tqdm

from data.const import NUM_CLASSES
from data.rasterize import rasterize_map
from evaluation.AP import instance_mask_AP
from evaluation.iou import get_batch_iou
from evaluate_json import SAMPLED_RECALLS, THRESHOLDS
from postprocess.vectorize import vectorize


def gen_dx_bx(xbound, ybound):
    dx = [row[2] for row in [xbound, ybound]]
    bx = [row[0] + row[2] / 2.0 for row in [xbound, ybound]]
    return dx, bx


def _unpack_output(model_output):
    if isinstance(model_output, dict):
        return model_output['semantic'], model_output['embedding'], model_output['direction']
    return model_output


def build_output_prefix(args):
    checkpoint_name = "in_memory"
    if getattr(args, 'modelf', None):
        checkpoint_name = os.path.splitext(os.path.basename(args.modelf))[0]
    return os.path.join(
        args.logdir,
        f"{args.experiment_name}_{args.branch_mode}_{args.version.split('-')[-1]}_{checkpoint_name}",
    )


def evaluate_semantic_iou(model, test_loader, args):
    model.eval()
    total_intersects = 0
    total_union = 0
    with torch.no_grad():
        for imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, semantic_gt, instance_gt, direction_gt, prior_map in tqdm.tqdm(test_loader):
            semantic, embedding, direction = _unpack_output(
                model(
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
                    prior_map.cuda(),
                    branch_mode=args.branch_mode,
                    return_branch_features=args.return_branch_features,
                )
            )
            semantic_gt = semantic_gt.cuda().float()
            intersects, union = get_batch_iou(_onehot_encoding(semantic), semantic_gt)
            total_intersects += intersects
            total_union += union
    iou = total_intersects / (total_union + 1e-7)
    miou = torch.mean(iou[1:]).item()
    return iou, miou


def _onehot_encoding(logits, dim=1):
    max_idx = torch.argmax(logits, dim, keepdim=True)
    one_hot = logits.new_full(logits.shape, 0)
    one_hot.scatter_(dim, max_idx, 1)
    return one_hot


def export_predictions_and_compute_map(model, test_loader, args):
    patch_h = args.ybound[1] - args.ybound[0]
    patch_w = args.xbound[1] - args.xbound[0]
    canvas_h = int(patch_h / args.ybound[2])
    canvas_w = int(patch_w / args.xbound[2])
    patch_size = (patch_h, patch_w)
    canvas_size = (canvas_h, canvas_w)
    dx, bx = gen_dx_bx(args.xbound, args.ybound)

    submission = {
        "meta": {
            "use_camera": True,
            "use_lidar": False,
            "use_radar": False,
            "use_external": False,
            "vector": True,
        },
        "results": {},
    }

    total_intersect = torch.zeros(NUM_CLASSES, device="cuda")
    total_union = torch.zeros(NUM_CLASSES, device="cuda")
    ap_matrix = torch.zeros((NUM_CLASSES, len(THRESHOLDS)), device="cuda")
    ap_count_matrix = torch.zeros((NUM_CLASSES, len(THRESHOLDS)), device="cuda")

    model.eval()
    with torch.no_grad():
        for batchi, batch in enumerate(tqdm.tqdm(test_loader)):
            imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, semantic_gt, instance_gt, direction_gt, prior_map = batch
            semantic, embedding, direction = _unpack_output(
                model(
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
                    prior_map.cuda(),
                    branch_mode=args.branch_mode,
                    return_branch_features=args.return_branch_features,
                )
            )

            for si in range(semantic.shape[0]):
                coords, confidences, line_types = vectorize(
                    semantic[si], embedding[si], direction[si], args.angle_class
                )
                vectors = []
                for coord, confidence, line_type in zip(coords, confidences, line_types):
                    vectors.append(
                        {
                            'pts': (coord * dx + bx).tolist(),
                            'pts_num': len(coord),
                            'type': line_type,
                            'confidence_level': confidence,
                        }
                    )

                sample_index = batchi * test_loader.batch_size + si
                rec = test_loader.dataset.samples[sample_index]
                submission['results'][rec['token']] = vectors

                pred_map, confidence_level = rasterize_map(
                    vectors, patch_size, canvas_size, NUM_CLASSES, args.thickness
                )
                gt_vectors = test_loader.dataset.get_vectors(rec)
                gt_map, _ = rasterize_map(
                    gt_vectors, patch_size, canvas_size, NUM_CLASSES, args.thickness
                )

                pred_map_tensor = torch.tensor(pred_map, device="cuda").unsqueeze(0)
                gt_map_tensor = torch.tensor(gt_map, device="cuda").unsqueeze(0)
                intersects, union = get_batch_iou(pred_map_tensor, gt_map_tensor)
                total_intersect += intersects.to(total_intersect.device)
                total_union += union.to(total_union.device)

                padded_confidence = confidence_level + [-1] * (300 - len(confidence_level))
                confidence_tensor = torch.tensor(padded_confidence, device="cuda").unsqueeze(0)
                instance_mask_AP(
                    ap_matrix,
                    ap_count_matrix,
                    pred_map_tensor,
                    gt_map_tensor,
                    args.xbound[2],
                    args.ybound[2],
                    confidence_tensor,
                    THRESHOLDS,
                    sampled_recalls=SAMPLED_RECALLS,
                )

    raster_iou = (total_intersect / (total_union + 1e-7)).mean().item()
    average_precision = torch.where(
        ap_count_matrix > 0,
        ap_matrix / ap_count_matrix,
        torch.full_like(ap_matrix, float("nan")),
    )
    map_value = torch.nanmean(average_precision).item()

    output_prefix = build_output_prefix(args)
    result_json_path = f"{output_prefix}_vectors.json"
    metrics_json_path = f"{output_prefix}_metrics.json"

    with open(result_json_path, "w") as handle:
        json.dump(submission, handle)
    with open(metrics_json_path, "w") as handle:
        json.dump(
            {
                "raster_iou": raster_iou,
                "map": map_value,
                "average_precision_matrix": average_precision.detach().cpu().tolist(),
            },
            handle,
            indent=2,
        )

    return {
        "raster_iou": raster_iou,
        "map": map_value,
        "result_json_path": result_json_path,
        "metrics_json_path": metrics_json_path,
    }


def run_full_evaluation(model, test_loader, args, logger=None):
    iou, miou = evaluate_semantic_iou(model, test_loader, args)
    map_metrics = export_predictions_and_compute_map(model, test_loader, args)
    with open(map_metrics['metrics_json_path'], "w") as handle:
        json.dump(
            {
                "miou": miou,
                "map": map_metrics['map'],
                "raster_iou": map_metrics['raster_iou'],
            },
            handle,
            indent=2,
        )
    message = (
        f"FINAL TEST[{args.experiment_name}]: "
        f"mIoU={miou:.4f} "
        f"mAP={map_metrics['map']:.4f} "
        f"vectors={map_metrics['result_json_path']} "
        f"metrics={map_metrics['metrics_json_path']}"
    )
    if logger is not None:
        logger.info(message)
    else:
        print(message)
    return {
        "iou": iou,
        "miou": miou,
        **map_metrics,
    }
