from __future__ import annotations

import argparse
import shutil
import sys
import traceback
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    default_weights_dir,
    ensure_dir,
    find_repo_root,
    read_csv,
    resolve_artifact_path,
    write_csv,
)
from sana06b_adapter import Sana06BAdapter  # noqa: E402


def infer_experiment_root(manifest: Path) -> Path:
    if manifest.parent.name == "manifests":
        return manifest.parent.parent
    return manifest.parent


def sync_paths(paths: list[Path], *, experiment_root: Path, drive_root: Path) -> None:
    for src in paths:
        if not src.exists():
            continue
        rel = src.resolve().relative_to(experiment_root.resolve())
        dst = drive_root / rel
        ensure_dir(dst.parent)
        shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--experiment-root", default=None)
    parser.add_argument("--backend", default=None, choices=["native", "diffusers"])
    parser.add_argument("--weights-dir", default=None)
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--sync_drive", action="store_true")
    parser.add_argument("--drive-experiment-root", default=None)
    args = parser.parse_args()

    repo_root = find_repo_root()
    manifest_path = Path(args.manifest).resolve()
    experiment_root = Path(args.experiment_root).resolve() if args.experiment_root else infer_experiment_root(manifest_path)
    rows = read_csv(manifest_path)
    if not rows:
        raise SystemExit(f"Manifest has no rows: {manifest_path}")
    fieldnames = list(rows[0].keys())

    backend = args.backend or rows[0].get("backend") or "native"
    adapter = Sana06BAdapter(
        backend=backend,
        repo_root=repo_root,
        weights_dir=args.weights_dir or default_weights_dir(),
        config_path=args.config_path,
        device=args.device,
        dtype=args.dtype,
        experiment_root=experiment_root,
    )
    drive_root = Path(args.drive_experiment_root).resolve() if args.drive_experiment_root else None

    for idx, row in enumerate(rows):
        output_path = resolve_artifact_path(row["output_path"], experiment_root)
        schedule_path = resolve_artifact_path(row["schedule_json_path"], experiment_root)
        runtime_path = resolve_artifact_path(row["runtime_json_path"], experiment_root)
        stdout_path = resolve_artifact_path(row["stdout_log_path"], experiment_root)
        stderr_path = resolve_artifact_path(row["stderr_log_path"], experiment_root)
        ensure_dir(stdout_path.parent)
        ensure_dir(stderr_path.parent)

        if args.resume and not args.force and row.get("status") == "completed" and output_path.exists():
            print(f"skip completed {row['run_id']}")
            continue
        if args.resume and not args.force and output_path.exists() and schedule_path.exists():
            row["status"] = "completed"
            row["error_message"] = ""
            write_csv(manifest_path, rows, fieldnames)
            print(f"skip existing {row['run_id']}")
            continue

        print(f"run {idx + 1}/{len(rows)} {row['run_id']}")
        try:
            runtime = adapter.run_one_case(row)
            stdout_path.write_text(f"completed {row['run_id']}\n{runtime}\n", encoding="utf-8")
            stderr_path.write_text("", encoding="utf-8")
            row["status"] = "completed"
            row["error_message"] = ""
        except Exception as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"
            row["status"] = "failed"
            row["error_message"] = err
            stderr_path.write_text(traceback.format_exc(), encoding="utf-8")
            stdout_path.write_text(f"failed {row['run_id']}\n", encoding="utf-8")
            write_csv(manifest_path, rows, fieldnames)
            print(err)
            raise
        write_csv(manifest_path, rows, fieldnames)
        if args.sync_drive:
            if drive_root is None:
                import os

                drive_env = os.environ.get("DRIVE_EXPERIMENT_ROOT")
                if not drive_env:
                    raise RuntimeError("--sync_drive requires --drive-experiment-root or DRIVE_EXPERIMENT_ROOT")
                drive_root = Path(drive_env).resolve()
            sync_paths(
                [output_path, schedule_path, runtime_path, stdout_path, stderr_path, manifest_path],
                experiment_root=experiment_root,
                drive_root=drive_root,
            )


if __name__ == "__main__":
    main()
