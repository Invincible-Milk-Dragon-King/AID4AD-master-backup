import argparse
from pathlib import Path

from branch_experiments.report import load_records, summarize_branch_metrics, write_metrics_csv


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize branch-level experiment metrics.")
    parser.add_argument("input", type=Path, help="JSON file containing per-model raw experiment scores.")
    parser.add_argument("output", type=Path, help="CSV file to write the summarized metrics to.")
    return parser.parse_args()


def main():
    args = parse_args()
    records = load_records(args.input)
    rows = summarize_branch_metrics(records)
    write_metrics_csv(args.output, rows)


if __name__ == "__main__":
    main()
