#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Batch run camera degradation probes.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["hdmapnet", "maptr", "admap", "gemap", "himap"],
        help="Model list",
    )
    parser.add_argument("--checkpoint-camera", required=True)
    parser.add_argument("--checkpoint-fusion", required=True)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--nproc-per-node", type=int, default=2)
    parser.add_argument("--master-port-base", type=int, default=29520)
    parser.add_argument("--registry", default=str(Path(__file__).with_name("model_registry.json")))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    runner = Path(__file__).with_name("run_probe.py")
    job_index = 0
    for model in args.models:
        for source, ckpt in (
            ("camera", args.checkpoint_camera),
            ("fusion", args.checkpoint_fusion),
        ):
            port = args.master_port_base + job_index
            cmd = [
                sys.executable,
                str(runner),
                "--model",
                model,
                "--source",
                source,
                "--checkpoint",
                ckpt,
                "--gpus",
                args.gpus,
                "--nproc-per-node",
                str(args.nproc_per_node),
                "--master-port",
                str(port),
                "--registry",
                args.registry,
            ]
            if args.dry_run:
                cmd.append("--dry-run")
            print(f"[RUN] model={model}, source={source}, port={port}")
            subprocess.run(cmd, check=True)
            job_index += 1


if __name__ == "__main__":
    main()
