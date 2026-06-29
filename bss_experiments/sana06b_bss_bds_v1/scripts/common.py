from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


EXPERIMENT_ID = "sana06b_bss_bds_v1"
MODEL_ID = "Sana-0.6B"
MODEL_VARIANT = "Sana_600M_1024px"
HF_PTH_REPO_ID = "Efficient-Large-Model/Sana_600M_1024px"
HF_DIFFUSERS_REPO_ID = "Efficient-Large-Model/Sana_600M_1024px_diffusers"
DEFAULT_CONFIG_PATH = "configs/sana_config/1024ms/Sana_600M_img1024.yaml"
DEFAULT_REFERENCE_NFE = 50
DEFAULT_FLOW_SHIFT = 4.0
DEFAULT_GUIDANCE_SCALE = 5.0
DEFAULT_PAG_GUIDANCE_SCALE = 2.0
DEFAULT_HEIGHT = 1024
DEFAULT_WIDTH = 1024


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def find_repo_root(start: Path | None = None) -> Path:
    path = (start or Path.cwd()).resolve()
    for candidate in [path, *path.parents]:
        if (candidate / ".git").exists():
            return candidate
    raise RuntimeError(f"Could not find repo root from {path}")


def experiment_dir(repo_root: Path | None = None) -> Path:
    root = repo_root or find_repo_root()
    return root / "bss_experiments" / EXPERIMENT_ID


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def prompt_hash(prompt: str) -> str:
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12]


def run_cmd(args: list[str], cwd: Path | None = None, timeout: int = 20) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False)
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except Exception as exc:  # noqa: BLE001
        return 999, "", f"{type(exc).__name__}: {exc}"


def git_info(repo_root: Path | None = None) -> dict[str, str]:
    root = repo_root or find_repo_root()
    branch = run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"], root)[1] or "unknown"
    commit = run_cmd(["git", "rev-parse", "HEAD"], root)[1] or "unknown"
    status = run_cmd(["git", "status", "--short"], root)[1]
    remotes = run_cmd(["git", "remote", "-v"], root)[1]
    return {
        "branch": branch,
        "commit": commit,
        "dirty_status": status if status else "clean",
        "remote": remotes.replace("\n", "; ") if remotes else "unknown",
    }


def official_time_uniform_flow_schedule(
    num_steps: int,
    *,
    t_start: float = 1.0,
    t_end: float = 0.001,
    flow_shift: float = DEFAULT_FLOW_SHIFT,
) -> list[float]:
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    betas = [t_start + (t_end - t_start) * i / num_steps for i in range(num_steps + 1)]
    sigmas = [1.0 - beta for beta in betas]
    shifted = [(flow_shift * sigma) / (1.0 + (flow_shift - 1.0) * sigma) for sigma in sigmas]
    return list(reversed(shifted))


def normalize_split_interval(index: int, num_intervals: int) -> int:
    idx = index if index >= 0 else num_intervals + index
    if idx < 0 or idx >= num_intervals:
        raise ValueError(f"split interval {index} is out of range for {num_intervals} intervals")
    return idx


def build_bss_flow_schedule(
    actual_nfe: int,
    *,
    split_pairs: tuple[int, int] = (0, -1),
    flow_shift: float = DEFAULT_FLOW_SHIFT,
) -> tuple[list[float], list[float], list[dict[str, float | int]]]:
    if actual_nfe < 4:
        raise ValueError("BSS needs at least 4 evaluations to split first and last intervals")
    base_steps = actual_nfe - 2
    base_coords = official_time_uniform_flow_schedule(base_steps, flow_shift=flow_shift)
    num_intervals = len(base_coords) - 1
    split_indices = sorted({normalize_split_interval(idx, num_intervals) for idx in split_pairs})

    final_coords: list[float] = [base_coords[0]]
    inserted: list[dict[str, float | int]] = []
    for interval_idx in range(num_intervals):
        left = base_coords[interval_idx]
        right = base_coords[interval_idx + 1]
        if interval_idx in split_indices:
            midpoint = 0.5 * (left + right)
            final_coords.append(midpoint)
            inserted.append(
                {
                    "interval_index": interval_idx,
                    "left": left,
                    "right": right,
                    "midpoint": midpoint,
                }
            )
        final_coords.append(right)
    return base_coords, final_coords, inserted


def schedule_payload(
    *,
    manifest_row: dict[str, str],
    sampler_mode: str,
    final_coords: list[float],
    base_coords: list[float] | None,
    inserted_midpoints: list[dict[str, float | int]] | None,
    git: dict[str, str] | None = None,
) -> dict:
    actual_nfe = int(manifest_row["actual_nfe"])
    row_git = git or git_info()
    return {
        "model_id": MODEL_ID,
        "model_variant": MODEL_VARIANT,
        "task": "t2i",
        "backend": manifest_row.get("backend", "native"),
        "method": manifest_row["method"],
        "sampler_mode": sampler_mode,
        "actual_nfe": actual_nfe,
        "model_eval_count": len(final_coords) - 1,
        "base_sample_steps": int(manifest_row["base_sample_steps"] or actual_nfe),
        "num_inference_steps": int(manifest_row["num_inference_steps"]),
        "split_pairs": manifest_row.get("split_pairs", ""),
        "reference_method": manifest_row.get("reference_method", "reference_uniform50"),
        "reference_nfe": int(manifest_row.get("reference_nfe") or DEFAULT_REFERENCE_NFE),
        "coordinate_type": "native_flow_dpm_solver_time_uniform_flow_shifted_sigma",
        "scheduler_name": manifest_row.get("scheduler_name", "flow_dpm-solver"),
        "flow_shift": float(manifest_row.get("flow_shift") or DEFAULT_FLOW_SHIFT),
        "base_coords": base_coords or final_coords,
        "final_coords": final_coords,
        "inserted_midpoints": inserted_midpoints or [],
        "seed": int(manifest_row["seed"]),
        "prompt_hash": manifest_row["prompt_hash"],
        "height": int(manifest_row["height"]),
        "width": int(manifest_row["width"]),
        "guidance_scale": float(manifest_row["guidance_scale"]),
        "pag_guidance_scale": float(manifest_row.get("pag_guidance_scale") or 0.0),
        "output_path": manifest_row["output_path"],
        "git_commit": row_git["commit"],
        "dirty_status": row_git["dirty_status"],
        "timestamp": utc_now_iso(),
    }


def validate_schedule_payload(payload: dict) -> list[str]:
    errors: list[str] = []
    final_coords = payload.get("final_coords") or []
    actual_nfe = int(payload.get("actual_nfe", -1))
    if len(final_coords) - 1 != actual_nfe:
        errors.append(f"expected {actual_nfe} intervals, got {len(final_coords) - 1}")
    if int(payload.get("model_eval_count", -1)) != actual_nfe:
        errors.append(f"model_eval_count is not {actual_nfe}")
    if payload.get("sampler_mode") == "bss":
        base_steps = int(payload.get("base_sample_steps", -1))
        if base_steps != actual_nfe - 2:
            errors.append(f"BSS base_sample_steps should be {actual_nfe - 2}, got {base_steps}")
        inserted = payload.get("inserted_midpoints") or []
        if len(inserted) != 2:
            errors.append(f"BSS should insert two midpoints, got {len(inserted)}")
    return errors


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_artifact_path(value: str | Path, experiment_root: Path) -> Path:
    value = str(value)
    expanded = os.path.expandvars(os.path.expanduser(value))
    path = Path(expanded)
    if path.is_absolute():
        return path
    return experiment_root / path


def write_markdown_table(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    ensure_dir(path.parent)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def default_weights_dir() -> str:
    drive_root = os.environ.get("DRIVE_WEIGHTS_ROOT")
    if drive_root:
        return str(Path(drive_root) / MODEL_VARIANT)
    return str(Path("/content/drive/MyDrive/ModelWeights/Sana") / MODEL_VARIANT)
