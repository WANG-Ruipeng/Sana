from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (  # noqa: E402
    ensure_dir,
    read_csv,
    resolve_artifact_path,
    write_csv,
)


def infer_experiment_root(manifest: Path) -> Path:
    return manifest.parent.parent if manifest.parent.name == "manifests" else manifest.parent


def load_rgb(path: Path):
    import numpy as np
    from PIL import Image

    img = Image.open(path).convert("RGB")
    return np.asarray(img).astype("float32") / 255.0


def image_metrics(image, ref) -> dict[str, float]:
    import numpy as np

    diff = image - ref
    l1 = float(np.mean(np.abs(diff)))
    l2 = float(np.sqrt(np.mean(diff * diff)))
    mse = float(np.mean(diff * diff))
    psnr = 99.0 if mse == 0 else float(20.0 * math.log10(1.0 / math.sqrt(mse)))
    return {"rgb_l1": l1, "rgb_l2": l2, "psnr": psnr}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--experiment-root", default=None)
    parser.add_argument("--metrics-dir", default=None)
    args = parser.parse_args()

    manifest = Path(args.manifest).resolve()
    experiment_root = Path(args.experiment_root).resolve() if args.experiment_root else infer_experiment_root(manifest)
    metrics_dir = Path(args.metrics_dir).resolve() if args.metrics_dir else experiment_root / "metrics"
    ensure_dir(metrics_dir)
    rows = [row for row in read_csv(manifest) if row.get("status") == "completed"]
    if not rows:
        print("No completed rows; writing empty metric files.")
        for name in ["master_long_metrics.csv", "per_case_metrics.csv", "same_compute_gain_long.csv"]:
            write_csv(metrics_dir / name, [], [])
        return

    by_case = {}
    for row in rows:
        by_case.setdefault(row["case_id"], []).append(row)

    metric_rows = []
    for case_id, case_rows in by_case.items():
        ref_rows = [r for r in case_rows if r["method"] == "reference_uniform50"]
        base_rows = [r for r in case_rows if r["method"] == "uniform8"]
        if not ref_rows or not base_rows:
            continue
        ref = load_rgb(resolve_artifact_path(ref_rows[0]["output_path"], experiment_root))
        baseline = load_rgb(resolve_artifact_path(base_rows[0]["output_path"], experiment_root))
        baseline_metrics = image_metrics(baseline, ref)
        baseline_l1 = baseline_metrics["rgb_l1"]
        for row in case_rows:
            image = load_rgb(resolve_artifact_path(row["output_path"], experiment_root))
            metrics = image_metrics(image, ref)
            closure = 1.0 if baseline_l1 == 0 else 1.0 - metrics["rgb_l1"] / baseline_l1
            runtime_seconds = ""
            runtime_path = resolve_artifact_path(row["runtime_json_path"], experiment_root)
            if runtime_path.exists():
                import json

                runtime_seconds = json.loads(runtime_path.read_text(encoding="utf-8")).get("seconds", "")
            metric_rows.append(
                {
                    "case_id": case_id,
                    "category": row.get("category", ""),
                    "prompt_hash": row["prompt_hash"],
                    "method": row["method"],
                    "method_family": row["method_family"],
                    "actual_nfe": row["actual_nfe"],
                    "reference_method": "reference_uniform50",
                    "reference_nfe": row["reference_nfe"],
                    "low_baseline_method": "uniform8",
                    "rgb_l1": f"{metrics['rgb_l1']:.8f}",
                    "rgb_l2": f"{metrics['rgb_l2']:.8f}",
                    "psnr": f"{metrics['psnr']:.6f}",
                    "rgb_l1_closure": f"{closure:.8f}",
                    "runtime_seconds": runtime_seconds,
                    "compute_fraction": f"{int(row['actual_nfe']) / int(row['reference_nfe']):.6f}",
                    "output_path": row["output_path"],
                }
            )

    fieldnames = [
        "case_id",
        "category",
        "prompt_hash",
        "method",
        "method_family",
        "actual_nfe",
        "reference_method",
        "reference_nfe",
        "low_baseline_method",
        "rgb_l1",
        "rgb_l2",
        "psnr",
        "rgb_l1_closure",
        "runtime_seconds",
        "compute_fraction",
        "output_path",
    ]
    write_csv(metrics_dir / "master_long_metrics.csv", metric_rows, fieldnames)
    write_csv(metrics_dir / "per_case_metrics.csv", metric_rows, fieldnames)

    lookup = {(row["case_id"], int(row["actual_nfe"]), row["method_family"]): row for row in metric_rows}
    gains = []
    for row in metric_rows:
        if row["method_family"] != "bss":
            continue
        key = (row["case_id"], int(row["actual_nfe"]), "uniform")
        uniform = lookup.get(key)
        if not uniform:
            continue
        gain = float(row["rgb_l1_closure"]) - float(uniform["rgb_l1_closure"])
        gains.append(
            {
                "case_id": row["case_id"],
                "actual_nfe": row["actual_nfe"],
                "bss_method": row["method"],
                "uniform_method": f"uniform{row['actual_nfe']}",
                "bss_rgb_l1_closure": row["rgb_l1_closure"],
                "uniform_rgb_l1_closure": uniform["rgb_l1_closure"],
                "gain": f"{gain:.8f}",
            }
        )
    write_csv(
        metrics_dir / "same_compute_gain_long.csv",
        gains,
        [
            "case_id",
            "actual_nfe",
            "bss_method",
            "uniform_method",
            "bss_rgb_l1_closure",
            "uniform_rgb_l1_closure",
            "gain",
        ],
    )
    print(f"Wrote metrics to {metrics_dir}")


if __name__ == "__main__":
    main()
