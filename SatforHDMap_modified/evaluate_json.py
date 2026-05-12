import json

import torch
import tqdm

from evaluation.dataset import HDMapNetEvalDataset
from evaluation.chamfer_distance import semantic_mask_chamfer_dist_cum
from evaluation.AP import instance_mask_AP
from evaluation.iou import get_batch_iou

SAMPLED_RECALLS = torch.linspace(0.1, 1, 10)
THRESHOLDS = [0.2, 0.5, 1.0]
CLASS_NAMES = ['divider', 'ped_crossing', 'boundary']


def _to_float_matrix(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().tolist()
    return [[float(item) for item in row] for row in value]


def _mean(values):
    values = [value for value in values if value == value]
    if len(values) == 0:
        return float('nan')
    return sum(values) / len(values)


def summarize_average_precision(average_precision, map_thresholds, class_names=None):
    class_names = class_names or CLASS_NAMES
    ap_matrix = _to_float_matrix(average_precision)
    thresholds = [float(threshold) for threshold in map_thresholds]

    per_class = {}
    for class_idx, row in enumerate(ap_matrix):
        class_name = class_names[class_idx] if class_idx < len(class_names) else f'class_{class_idx}'
        ap_by_threshold = {
            f'AP@{threshold:g}': row[threshold_idx]
            for threshold_idx, threshold in enumerate(thresholds)
        }
        class_map = _mean(row)
        per_class[class_name] = {
            **ap_by_threshold,
            'mAP': class_map,
            'mAP_percent': class_map * 100,
        }

    per_threshold = {}
    for threshold_idx, threshold in enumerate(thresholds):
        threshold_values = [row[threshold_idx] for row in ap_matrix]
        threshold_ap = _mean(threshold_values)
        per_threshold[f'AP@{threshold:g}'] = threshold_ap

    all_values = [value for row in ap_matrix for value in row]
    map_value = _mean(all_values)
    return {
        'class_names': class_names[:len(ap_matrix)],
        'map_thresholds': thresholds,
        'average_precision_matrix': ap_matrix,
        'per_class': per_class,
        'per_threshold': per_threshold,
        'mAP': map_value,
        'mAP_percent': map_value * 100,
    }


def serialize_eval_result(result):
    summary = summarize_average_precision(result['Average_precision'], result['map_thresholds'])
    serialized = {
        'average_precision': summary,
    }
    for key in ('iou', 'CD_pred', 'CD_label', 'CD'):
        if key in result:
            value = result[key]
            if torch.is_tensor(value):
                value = value.detach().cpu().tolist()
            serialized[key] = value
    return serialized


def get_val_info(args):
    map_thresholds = getattr(args, 'map_thresholds', None) or THRESHOLDS
    data_conf = {
        'xbound': args.xbound,
        'ybound': args.ybound,
        'thickness': args.thickness,
    }

    dataset = HDMapNetEvalDataset(
        args.version,
        args.dataroot,
        args.eval_set,
        args.result_path,
        data_conf,
        prior_map_root=args.prior_map_root,
        satellite_map_size=tuple(args.satellite_map_size),
        is_newsplit=args.is_newsplit,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.bsz,
        shuffle=False,
        drop_last=False,
        num_workers=args.nworkers,
    )

    if not args.ap_only:
        total_CD1 = torch.zeros(args.max_channel).cuda()
        total_CD2 = torch.zeros(args.max_channel).cuda()
        total_CD_num1 = torch.zeros(args.max_channel).cuda()
        total_CD_num2 = torch.zeros(args.max_channel).cuda()
        total_intersect = torch.zeros(args.max_channel).cuda()
        total_union = torch.zeros(args.max_channel).cuda()
    AP_matrix = torch.zeros((args.max_channel, len(map_thresholds))).cuda()
    AP_count_matrix = torch.zeros((args.max_channel, len(map_thresholds))).cuda()

    print('running eval...')
    for pred_map, confidence_level, gt_map in tqdm.tqdm(data_loader):
        # iou
        pred_map = pred_map.cuda()
        confidence_level = confidence_level.cuda()
        gt_map = gt_map.cuda()

        if not args.ap_only:
            intersect, union = get_batch_iou(pred_map, gt_map)
            CD1, CD2, num1, num2 = semantic_mask_chamfer_dist_cum(pred_map, gt_map, args.xbound[2], args.ybound[2], threshold=args.CD_threshold)

        instance_mask_AP(AP_matrix, AP_count_matrix, pred_map, gt_map, args.xbound[2], args.ybound[2],
                         confidence_level, map_thresholds, sampled_recalls=SAMPLED_RECALLS)

        if not args.ap_only:
            total_intersect += intersect.cuda()
            total_union += union.cuda()
            total_CD1 += CD1
            total_CD2 += CD2
            total_CD_num1 += num1
            total_CD_num2 += num2

    result = {
        'Average_precision': AP_matrix / AP_count_matrix,
        'map_thresholds': map_thresholds,
    }
    if not args.ap_only:
        CD_pred = total_CD1 / total_CD_num1
        CD_label = total_CD2 / total_CD_num2
        CD = (total_CD1 + total_CD2) / (total_CD_num1 + total_CD_num2)
        CD_pred[CD_pred > args.CD_threshold] = args.CD_threshold
        CD_label[CD_label > args.CD_threshold] = args.CD_threshold
        CD[CD > args.CD_threshold] = args.CD_threshold
        result.update({
            'iou': total_intersect / total_union,
            'CD_pred': CD_pred,
            'CD_label': CD_label,
            'CD': CD,
        })
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate nuScenes local HD Map Construction Results.')
    parser.add_argument('--result_path', type=str)
    parser.add_argument('--dataroot', type=str, default='dataset/nuScenes/')
    parser.add_argument('--prior_map_root', type=str, default='./satmap/prior_map_trainval')
    parser.add_argument('--bsz', type=int, default=4)
    parser.add_argument('--nworkers', type=int, default=0)
    parser.add_argument('--version', type=str, default='v1.0-mini', choices=['v1.0-trainval', 'v1.0-mini', 'v1.0-test'])
    parser.add_argument('--eval_set', type=str, default='mini_val', choices=['train', 'val', 'test', 'mini_train', 'mini_val'])
    parser.add_argument('--is_newsplit', action='store_true')
    parser.add_argument('--ap_only', action='store_true')
    parser.add_argument('--map_thresholds', nargs='+', type=float, default=None)
    parser.add_argument('--thickness', type=int, default=5)
    parser.add_argument('--max_channel', type=int, default=3)
    parser.add_argument('--CD_threshold', type=int, default=5)
    parser.add_argument('--satellite_map_size', nargs=2, type=int, default=[400, 200])
    parser.add_argument('--output_json', type=str, default=None)
    parser.add_argument("--xbound", nargs=3, type=float, default=[-30.0, 30.0, 0.15])
    parser.add_argument("--ybound", nargs=3, type=float, default=[-15.0, 15.0, 0.15])

    args = parser.parse_args()

    result = get_val_info(args)
    serialized = serialize_eval_result(result)
    print(json.dumps(serialized, indent=2))
    if args.output_json is not None:
        with open(args.output_json, 'w') as handle:
            json.dump(serialized, handle, indent=2)
        print(f'saved metrics to {args.output_json}')
