from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import ensure_dir, read_csv, resolve_artifact_path  # noqa: E402


def infer_experiment_root(manifest: Path) -> Path:
    return manifest.parent.parent if manifest.parent.name == "manifests" else manifest.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--metrics-dir", default=None)
    parser.add_argument("--figures-dir", default=None)
    parser.add_argument("--experiment-root", default=None)
    args = parser.parse_args()

    manifest = Path(args.manifest).resolve()
    experiment_root = Path(args.experiment_root).resolve() if args.experiment_root else infer_experiment_root(manifest)
    metrics_dir = Path(args.metrics_dir).resolve() if args.metrics_dir else experiment_root / "metrics"
    figures_dir = Path(args.figures_dir).resolve() if args.figures_dir else experiment_root / "figures"
    ensure_dir(figures_dir)

    metrics_path = metrics_dir / "master_long_metrics.csv"
    if metrics_path.exists() and metrics_path.stat().st_size:
        import matplotlib.pyplot as plt

        rows = read_csv(metrics_path)
        series = {"uniform": {}, "bss": {}}
        for family in series:
            for nfe in sorted({int(r["actual_nfe"]) for r in rows if r["method_family"] == family}):
                values = [float(r["rgb_l1_closure"]) for r in rows if r["method_family"] == family and int(r["actual_nfe"]) == nfe]
                if values:
                    series[family][nfe] = sum(values) / len(values)
        plt.figure(figsize=(6, 4))
        for family, values in series.items():
            if not values:
                continue
            xs = [nfe / 50.0 for nfe in values]
            ys = [values[nfe] for nfe in values]
            plt.plot(xs, ys, marker="o", label=family)
        plt.xlabel("NFE / reference NFE")
        plt.ylabel("RGB L1 closure")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / "compute_quality_rgb_closure.png", dpi=180)
        plt.savefig(figures_dir / "compute_quality_rgb_closure.pdf")
        plt.close()
    else:
        (figures_dir / "compute_quality_rgb_closure.placeholder.txt").write_text(
            "No completed metrics available yet.\n", encoding="utf-8"
        )

    rows = [row for row in read_csv(manifest) if row.get("status") == "completed"]
    html_dir = figures_dir / "side_by_side"
    ensure_dir(html_dir)
    by_case = {}
    for row in rows:
        by_case.setdefault(row["case_id"], []).append(row)
    lines = [
        "<!doctype html><meta charset='utf-8'><title>Sana-0.6B side by side</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px} img{width:180px;max-width:22vw} .grid{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:28px}.cell{width:180px}</style>",
        "<h1>Sana-0.6B side by side</h1>",
    ]
    for case_id, case_rows in by_case.items():
        lines.append(f"<h2>{html.escape(case_id)}</h2><div class='grid'>")
        preferred = ["uniform8", "uniform10", "bss10", "uniform20", "bss20", "uniform30", "bss30", "uniform40", "bss40", "reference_uniform50"]
        lookup = {row["method"]: row for row in case_rows}
        for method in preferred:
            row = lookup.get(method)
            if not row:
                continue
            img_path = resolve_artifact_path(row["output_path"], experiment_root)
            rel = Path("../..") / img_path.relative_to(experiment_root)
            lines.append(f"<div class='cell'><img src='{html.escape(str(rel).replace(chr(92), '/'))}'><br>{html.escape(method)}</div>")
        lines.append("</div>")
    if not by_case:
        lines.append("<p>No completed images available yet.</p>")
    (html_dir / "index.html").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote figures to {figures_dir}")


if __name__ == "__main__":
    main()
