"""Offline environment check and explicit read-only device capture, without LAWM."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

from lawm_deploy import config
from lawm_deploy.hardware_contract import REQUIRED_SITE_CHECKS, hand_contract


def offline(cfg):
    result = dict(python=sys.executable, version=platform.python_version(), packages={},
                  hardware_connected=False, site_checks=cfg.get("commissioning", {}))
    for name in ("numpy", "scipy", "mujoco", "yaml", "viser", "rtde_receive", "rtde_control",
                 "pyrealsense2", "sharpa", "torch", "torch_scatter", "spconv.pytorch", "timm", "lawm"):
        try:
            module = importlib.import_module(name)
            result["packages"][name] = dict(ok=True, version=str(getattr(module, "__version__", "native/unversioned")),
                                            file=str(getattr(module, "__file__", "")))
        except Exception as error:
            result["packages"][name] = dict(ok=False, error=str(error))
    from lawm_deploy.robot_model import cell_spec
    model = cell_spec(config.mount_pose(cfg)).compile()
    result["compiled_robot"] = dict(nq=model.nq, nv=model.nv, expected_hand_joints=len(hand_contract()[0]))
    checkpoint = config.repo_path(cfg["model"]["checkpoint"])
    sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    with np.load(config.repo_path(cfg["model"]["prior"]), allow_pickle=False) as prior:
        result["checkpoint"] = dict(sha256=sha, prior_matches=str(prior["checkpoint_sha256"]) == sha,
                                    prior_shape=list(prior["z_mean"].shape))
    if result["packages"]["torch"]["ok"]:
        import torch
        result["cuda"] = dict(available=torch.cuda.is_available(), build=torch.version.cuda)
        if torch.cuda.is_available():
            result["cuda"]["device"] = torch.cuda.get_device_name(0)
            result["cuda"]["tensor_check"] = float((torch.ones(8, device="cuda") * 2).sum()) == 16
    result["pending_site_checks"] = [k for k in REQUIRED_SITE_CHECKS if cfg.get("commissioning", {}).get(k) is not True]
    result["model_environment_ready"] = all(p["ok"] for p in result["packages"].values()) and result["checkpoint"]["prior_matches"]
    result["physical_validation"] = "NOT_PERFORMED"
    return result


def observe(cfg, *, cameras=False, duration=3.0, output=None):
    """Read-only feedback, optionally calibrated depth. Never creates RTDEControl."""
    from lawm_deploy.cell import RealCell
    output = Path(output) if output else config.repo_path(cfg["log_dir"]) / time.strftime("preflight_%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=True)
    cell, rows, result = None, [], dict(mode="read_only", cameras=cameras, completed=False)
    try:
        cell = RealCell(cfg["cell"], hand_contract()[0], cameras=cameras, full_config=cfg)
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            q, h, tcp = cell.state()
            rows.append(dict(receipt_monotonic=time.monotonic(), arm_q=q.tolist(), hand_q=h.tolist(),
                             reported_tcp=tcp.tolist(), tcp_offset_verified=False))
            if cameras:
                views = cell.views()
                qa, qh, _ = cell.state()
                if np.max(np.abs(qa-q)) > 0.01 or np.max(np.abs(qh-h)) > 0.03:
                    raise RuntimeError("Robot moved while capturing depth; no paired sample saved")
                arrays = {}
                for i, (depth, K, T) in enumerate(views):
                    arrays.update({f"depth_{i}": depth, f"K_{i}": K, f"T_base_depth_{i}": T})
                    raw, scale = cell.cameras.last_raw_depth[i]
                    arrays.update({f"sensor_depth_z16_{i}": raw, f"sensor_depth_scale_m_{i}": scale})
                arrays.update(q_arm=q, q_hand=h, reported_tcp=tcp, config_json=json.dumps(cfg),
                              camera_metadata_json=json.dumps(cell.cameras.metadata),
                              observation_info_json=json.dumps(dict(camera_frames=cell.cameras.last_frames,
                                  source="read_only_preflight; no FK hand or processed model input")))
                np.savez_compressed(output / f"depth_{len(rows):03d}.npz", **arrays)
                rows[-1]["camera_frames"] = cell.cameras.last_frames
            time.sleep(0.05)
        result.update(completed=True, samples=len(rows),
                      camera_metadata=cell.cameras.metadata if cell.cameras else [],
                      note="Read-only queries only; TCP offset and physical motion have not been validated")
    except BaseException as error:
        result["error"] = str(error)
        raise
    finally:
        if cell is not None:
            cell.close()
        (output / "feedback.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        (output / "status.json").write_text(json.dumps(result, indent=2) + "\n")
        print(f"Read-only report: {output}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/real.yaml")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--devices", action="store_true", help="connect UR receive and Sharpa feedback only; no cameras")
    mode.add_argument("--cameras", action="store_true", help="read-only devices plus calibrated depth; requires calibration.yaml")
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--output", help="offline JSON file, or device-capture directory")
    args = parser.parse_args()
    cfg = config.load(args.config)
    if args.devices or args.cameras:
        if not np.isfinite(args.duration) or not 0 < args.duration <= 60:
            parser.error("duration must be in (0, 60] seconds")
        observe(cfg, cameras=args.cameras, duration=args.duration, output=args.output)
    else:
        result = offline(cfg)
        text = json.dumps(result, indent=2)
        print(text)
        if args.output:
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text + "\n")
        if not result["model_environment_ready"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
