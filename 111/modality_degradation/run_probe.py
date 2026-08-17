#!/usr/bin/env python3
import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


CAMERA_SOURCES = ("camera", "fusion")
LIDAR_SOURCES = ("lidar", "fusion_lidar")
ALL_SOURCES = CAMERA_SOURCES + LIDAR_SOURCES


def load_registry(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_repo_root(repo_root_value: str, workspace_root: Path) -> Path:
    """Accept absolute paths or paths relative to 111/."""
    p = Path(repo_root_value)
    if p.is_absolute():
        return p
    return (workspace_root / p).resolve()


def build_cmd(
    cfg,
    model,
    source,
    checkpoint,
    gpus,
    nproc,
    master_port,
    workspace_root,
    global_batch=16,
):
    repo_root = resolve_repo_root(cfg["repo_root"], workspace_root)
    entry = cfg["entry_script"]
    launch_style = cfg.get("launch_style", "satfor_probe")
    probe_family = "lidar" if source in LIDAR_SOURCES else "camera"
    exp_name = f"{probe_family}_decoder_probe_{model}_{source}"
    logdir = f"{cfg.get('logdir_prefix', './branch_runs')}/{exp_name}"

    if launch_style == "satfor_probe":
        # Stock HDMapNet train.py is single-process; use first visible GPU.
        feature_map = cfg.get("source_to_feature") or {}
        if source not in feature_map:
            raise ValueError(
                f"Model {model} does not define source_to_feature[{source}]. "
                f"Available: {sorted(feature_map)}"
            )
        feature = feature_map[source]
        model_map = cfg.get("source_to_model") or {}
        model_name = model_map.get(
            source, cfg.get("default_model", "HDMapNet_cam")
        )
        branch_mode = "lidar_only" if source in LIDAR_SOURCES else "camera_only"
        probe_epochs = int(os.environ.get("PROBE_EPOCHS", cfg.get("probe_epochs", 24)))
        # -u: unbuffered so tee/logs show progress immediately
        base = [sys.executable, "-u", entry]
        base.extend(cfg.get("base_args", []))
        base.extend(
            [
                "--model",
                model_name,
                "--branch_mode",
                branch_mode,
                "--train_decoder_only",
                "--probe_feature_source",
                feature,
                "--probe_encoder_checkpoint",
                checkpoint,
                "--logdir",
                logdir,
                "--nepochs",
                str(probe_epochs),
            ]
        )
        samples_per_gpu = None
    elif launch_style == "mmdet_probe":
        if global_batch % nproc != 0:
            raise ValueError(
                f"global_batch={global_batch} must be divisible by nproc={nproc}"
            )
        samples_per_gpu = global_batch // nproc
        config_map = cfg.get("source_to_config") or {}
        if source not in config_map:
            raise ValueError(
                f"Model {model} does not define source_to_config[{source}]. "
                f"Available: {sorted(config_map)}"
            )
        config_path = config_map[source]
        cfg_options = list(cfg.get("cfg_options", []))
        cfg_options.append(f"data.samples_per_gpu={samples_per_gpu}")
        cfg_options.append(f"load_from={checkpoint}")
        # Decoder-only freezes backbone/neck; DDP needs this when grads are sparse.
        probe_epochs = int(os.environ.get("PROBE_EPOCHS", cfg.get("probe_epochs", 24)))
        cfg_options.append(f"total_epochs={probe_epochs}")
        cfg_options.append(f"runner.max_epochs={probe_epochs}")
        if "find_unused_parameters=True" not in cfg_options:
            cfg_options.append("find_unused_parameters=True")
        # Camera probe from fusion: skip LiDAR/fuser.
        if source == "fusion":
            cfg_options.append("model.force_camera_only=True")
        # LiDAR probe from fusion: skip camera / fuser (lidar-only BEV path).
        if source == "fusion_lidar":
            cfg_options.append("model.force_lidar_only=True")
        # Guard common GeMap mistake: simple config + full checkpoint.
        if model == "gemap" and source == "camera":
            ckpt_name = Path(checkpoint).name.lower()
            if "full" in ckpt_name and "simple" not in ckpt_name:
                raise ValueError(
                    "GeMap camera probe must use gemap_simple_r50_110ep.pth with "
                    "gemap_simple_r50_110ep.py. You passed a full checkpoint: "
                    f"{checkpoint}. Download simple camera weights first."
                )
        config_arg = cfg.get("config_arg")
        decoder_keywords = cfg.get(
            "decoder_train_keywords",
            "pts_bbox_head",
        )
        # Encoder-side denylist: BEV encoder / fuser / camera+lidar encoding stay frozen.
        # Only transformer.decoder + query/cls/reg branches remain trainable.
        decoder_freeze_keywords = cfg.get(
            "decoder_freeze_keywords",
            (
                "transformer.encoder,transformer.fuser,bev_embedding,"
                "positional_encoding,level_embeds,cams_embeds,can_bus_mlp,"
                "reference_points,compress_layer,conv_mask_bev_embed,"
                "conv_bev_embed,sem_classifier,pts_bbox_head.backbone,"
                "pts_bbox_head.neck,lidar_modal"
            ),
        )

        # Same as tools/dist_train.sh: multi-GPU via torch.distributed.launch.
        base = [
            sys.executable,
            "-m",
            "torch.distributed.launch",
            "--nproc_per_node",
            str(nproc),
            "--master_port",
            str(master_port),
            entry,
        ]
        if config_arg:
            base.extend([config_arg, config_path])
        else:
            base.append(config_path)
        base.extend(
            [
                "--work-dir",
                logdir,
                "--launcher",
                "pytorch",
                "--decoder-only",
                "--decoder-train-keywords",
                decoder_keywords,
                "--decoder-freeze-keywords",
                decoder_freeze_keywords,
                "--cfg-options",
            ]
        )
        base.extend(cfg_options)
    else:
        raise ValueError(f"Unsupported launch_style: {launch_style}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpus
    env["PYTHONUNBUFFERED"] = "1"
    # Match dist_train/dist_test: repo root must be on PYTHONPATH so
    # `import projects` works when entry is tools/train.py (sys.path[0]=tools/).
    # HDMapNet (satfor_probe) does not need this and a polluted PYTHONPATH can
    # break `import torch` (NameError: _C is not defined).
    if launch_style == "mmdet_probe":
        repo_str = str(repo_root)
        prev = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = repo_str if not prev else f"{repo_str}{os.pathsep}{prev}"
    return repo_root, base, env, samples_per_gpu, global_batch


def main():
    parser = argparse.ArgumentParser(
        description="Run camera/lidar modality-degradation probes."
    )
    parser.add_argument("--model", required=True, help="hdmapnet/maptr/admap/gemap/himap")
    parser.add_argument(
        "--source",
        required=True,
        choices=list(ALL_SOURCES),
        help=(
            "camera|fusion for camera-branch probes; "
            "lidar|fusion_lidar for lidar-branch probes"
        ),
    )
    parser.add_argument("--checkpoint", required=True, help="Path to pretrained encoder checkpoint")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--nproc-per-node", type=int, default=2)
    parser.add_argument("--master-port", type=int, default=29506)
    parser.add_argument(
        "--global-batch",
        type=int,
        default=None,
        help="Global batch size = samples_per_gpu * nproc. "
        "Default: GLOBAL_BATCH env, else registry probe_global_batch*, else 16.",
    )
    parser.add_argument("--registry", default=str(Path(__file__).with_name("model_registry.json")))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    registry_path = Path(args.registry).resolve()
    # modality_degradation/ -> 111/
    workspace_root = registry_path.parent.parent
    registry = load_registry(registry_path)
    if args.model not in registry:
        raise ValueError(f"Unknown model: {args.model}")
    cfg = registry[args.model]
    if args.global_batch is not None:
        global_batch = args.global_batch
    elif os.environ.get("GLOBAL_BATCH"):
        global_batch = int(os.environ["GLOBAL_BATCH"])
    else:
        by_src = cfg.get("probe_global_batch_by_source") or {}
        global_batch = int(
            by_src.get(args.source, cfg.get("probe_global_batch", 16))
        )
    repo_root, cmd, env, samples_per_gpu, global_batch = build_cmd(
        cfg,
        args.model,
        args.source,
        args.checkpoint,
        args.gpus,
        args.nproc_per_node,
        args.master_port,
        workspace_root,
        global_batch=global_batch,
    )

    if not repo_root.exists():
        raise FileNotFoundError(f"repo_root does not exist: {repo_root}")

    n_visible = len([x for x in args.gpus.split(",") if x.strip() != ""])
    if args.nproc_per_node != n_visible:
        print(
            f"[WARN] nproc_per_node={args.nproc_per_node} != "
            f"visible GPU count={n_visible} ({args.gpus}); "
            "set NPROC to match CUDA_VISIBLE_DEVICES."
        )

    rendered = " ".join(shlex.quote(c) for c in cmd)
    print(f"[INFO] cwd={repo_root}", flush=True)
    print(f"[INFO] CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}", flush=True)
    print(f"[INFO] nproc_per_node={args.nproc_per_node}", flush=True)
    if samples_per_gpu is not None:
        print(
            f"[INFO] global_batch={global_batch}  "
            f"samples_per_gpu={samples_per_gpu}  "
            f"(={samples_per_gpu}×{args.nproc_per_node})",
            flush=True,
        )
    print(f"[CMD] {rendered}", flush=True)

    if args.dry_run:
        return

    # Preflight: catch broken torch installs / env before long data loading.
    preflight = subprocess.run(
        [sys.executable, "-c", "import torch; print(torch.__version__)"],
        env=env,
        capture_output=True,
        text=True,
    )
    if preflight.returncode != 0:
        print("[ERROR] Preflight `import torch` failed in launch env:", flush=True)
        print(preflight.stderr or preflight.stdout, flush=True)
        print(
            "[HINT] Ensure `conda activate hdmapnet` (or the model env) and "
            "that PYTHONPATH does not shadow torch.",
            flush=True,
        )
        raise SystemExit(preflight.returncode)

    # One retry: intermittent torch `_C` NameError has been observed under load.
    last_err = None
    for attempt in range(1, 3):
        try:
            subprocess.run(cmd, cwd=str(repo_root), env=env, check=True)
            last_err = None
            break
        except subprocess.CalledProcessError as err:
            last_err = err
            if attempt == 1:
                print(
                    f"[WARN] train attempt {attempt} failed (exit {err.returncode}); "
                    "retrying once...",
                    flush=True,
                )
    if last_err is not None:
        raise last_err


if __name__ == "__main__":
    main()
