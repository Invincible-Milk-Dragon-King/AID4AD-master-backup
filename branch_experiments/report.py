import csv
import json
from pathlib import Path

from .protocol import BranchMetricsInput, compute_branch_metrics


RAW_METRIC_FIELDS = [
    "camera_only",
    "fusion_base",
    "fusion_base_drop_satellite",
    "fusion_distilled",
    "fusion_distilled_drop_satellite",
]


def summarize_branch_metrics(records):
    rows = []
    for record in records:
        derived_metrics = compute_branch_metrics(
            BranchMetricsInput(
                camera_only=record["camera_only"],
                fusion_base=record["fusion_base"],
                fusion_base_drop_satellite=record["fusion_base_drop_satellite"],
                fusion_distilled=record["fusion_distilled"],
                fusion_distilled_drop_satellite=record["fusion_distilled_drop_satellite"],
            )
        )
        rows.append({
            "model": record["model"],
            "metric": record.get("metric", "score"),
            **{field: record[field] for field in RAW_METRIC_FIELDS},
            **derived_metrics,
        })
    return rows


def write_metrics_csv(output_path, rows):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "metric", *RAW_METRIC_FIELDS, "FusionGain", "CameraDegenerationGap", "RecoveryGain", "FinalPerformanceGain"]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        formatted_rows = []
        for row in rows:
            formatted_row = {}
            for key, value in row.items():
                if isinstance(value, float):
                    formatted_row[key] = round(value, 6)
                else:
                    formatted_row[key] = value
            formatted_rows.append(formatted_row)
        writer.writerows(formatted_rows)


def load_records(input_path):
    input_path = Path(input_path)
    with input_path.open() as handle:
        return json.load(handle)
