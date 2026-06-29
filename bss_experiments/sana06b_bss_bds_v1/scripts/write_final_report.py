from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import read_csv  # noqa: E402


def read_text(path: Path, fallback: str) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else fallback


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", required=True)
    args = parser.parse_args()
    root = Path(args.experiment_root).resolve()
    reports = root / "reports"
    tables = root / "tables"
    reports.mkdir(parents=True, exist_ok=True)

    cross_path = tables / "table_cross_model_same_compute_sana06b_row.csv"
    verdict = "Need more data"
    cases = "0"
    if cross_path.exists() and cross_path.stat().st_size:
        rows = read_csv(cross_path)
        if rows:
            verdict = rows[0].get("BDS", verdict)
            cases = rows[0].get("Cases", cases)

    text = f"""# Final Sana-0.6B BSS/BDS report

## Purpose

This is a Sana-0.6B T2I sanity check for same-compute BSS vs uniform sampling using a fixed prompt suite and official Sana code. It is not an official benchmark and makes no universal claim.

## Repo/model/hardware audit

See `reports/00_repo_model_hardware_audit.md`.

## Weight location and Drive setup

Weights are expected outside git, normally at `/content/drive/MyDrive/ModelWeights/Sana/Sana_600M_1024px`.

## Backend selected

Primary backend: native Sana pipeline. The runner uses `app.sana_pipeline.SanaPipeline` with `configs/sana_config/1024ms/Sana_600M_img1024.yaml`. Diffusers is a fallback only if the 0.6B diffusers checkpoint is loadable and exposes compatible timestep control.

## BSS schedule implementation summary

BSS-T is implemented as base `(T - 2)` official `time_uniform_flow` coordinates with the first and last intervals split. Uniform rows use official `time_uniform_flow` unchanged. Native BSS rows inject the custom schedule through a narrow runtime patch of `DPM_Solver.get_time_steps`; model weights and prompt conditioning are not modified.

## Smoke result

See `reports/01_sana06b_smoke_report.md`. If weights or GPU were missing, smoke is not run.

## Mini-suite completion summary

Completed prompt cases: {cases}. See `reports/02_mini_suite_run_report.md`.

## Same-compute RGB-L1 closure row

See `tables/table_cross_model_same_compute_sana06b_row.md` and `.tex`.

## BDS calibration/holdout result

See `tables/tableA_sana06b_bds_by_split.md` and `reports/03_bds_report.md`.

## Final verdict

{verdict}

## Include in cross-model paper table?

Include only after smoke and the fixed mini-suite complete with valid schedule JSON, metrics, and BDS tables. Until then, mark Sana-0.6B as Need more data.

## Caveats

- Fixed prompt suite, not an official benchmark.
- Reference is uniform50, not ground truth.
- NFE is used as a compute proxy.
- Backend/config mapping for 0.6B must be audited in the runtime environment.
- T2I has stochastic multi-solution behavior.

## Next steps

- Run smoke in Colab with Drive weights.
- Run the 16-prompt mini-suite only after smoke passes.
- Try Sana-1.5-1.6B if 0.6B is promising.
- Use PixArt fallback if Sana integration is too hard.
- Run solver-coordinate ablation if results are mixed.
"""
    out = reports / "FINAL_SANA06B_BSS_BDS_REPORT.md"
    out.write_text(text, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
