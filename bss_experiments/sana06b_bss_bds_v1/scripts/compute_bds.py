from __future__ import annotations

import argparse
import random
import statistics
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import ensure_dir, read_csv, write_csv, write_markdown_table  # noqa: E402


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def bootstrap_ci(values: list[float], iterations: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    if len(values) == 1:
        return values[0], values[0], values[0]
    rng = random.Random(seed)
    samples = []
    for _ in range(iterations):
        draw = [values[rng.randrange(len(values))] for _ in values]
        samples.append(mean(draw))
    samples.sort()
    lo = samples[int(0.025 * (len(samples) - 1))]
    hi = samples[int(0.975 * (len(samples) - 1))]
    lcb = samples[int(0.05 * (len(samples) - 1))]
    return lcb, lo, hi


def split_cases(cases: list[str]) -> dict[str, tuple[set[str], set[str]]]:
    ordered = sorted(cases)
    first_half_cal = set(ordered[: len(ordered) // 2])
    alternating_cal = {case for idx, case in enumerate(ordered) if idx % 2 == 0}
    rng = random.Random(0)
    shuffled = ordered[:]
    rng.shuffle(shuffled)
    random_cal = set(shuffled[: len(shuffled) // 2])
    return {
        "first_half_split": (first_half_cal, set(ordered) - first_half_cal),
        "alternating_split": (alternating_cal, set(ordered) - alternating_cal),
        "random_split_seed0": (random_cal, set(ordered) - random_cal),
    }


def verdict(low_lcb: float, all_lcb: float, cases: int) -> str:
    if cases < 8:
        return "Need more data"
    if all_lcb > 0:
        return "Green"
    if low_lcb > 0 and all_lcb <= 0:
        return "Low-only"
    if low_lcb <= 0:
        return "Reject"
    return "Need more data"


def format_cell(closure: float | None, gain: float | None, nfe: int) -> str:
    if closure is None or gain is None:
        return "--"
    return f"{closure:.3f} / {gain:+.3f} [{nfe} NFE]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-dir", required=True)
    parser.add_argument("--tables-dir", default=None)
    parser.add_argument("--reports-dir", default=None)
    parser.add_argument("--bootstrap-iters", type=int, default=2000)
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir).resolve()
    tables_dir = Path(args.tables_dir).resolve() if args.tables_dir else metrics_dir.parent / "tables"
    reports_dir = Path(args.reports_dir).resolve() if args.reports_dir else metrics_dir.parent / "reports"
    ensure_dir(tables_dir)
    ensure_dir(reports_dir)

    gains_path = metrics_dir / "same_compute_gain_long.csv"
    master_path = metrics_dir / "master_long_metrics.csv"
    gains = read_csv(gains_path) if gains_path.exists() and gains_path.stat().st_size else []
    master = read_csv(master_path) if master_path.exists() and master_path.stat().st_size else []

    if not gains:
        split_rows = []
        cross_row = {
            "Model": "Sana-0.6B",
            "Setting": "Image Generation",
            "Few-step prior": "None / not explicit",
            "Ref.": "50",
            "Cases": "0",
            "BDS": "Need more data",
            "Low": "--",
            "Middle Low": "--",
            "Middle High": "--",
            "High": "--",
            "Mean Delta": "--",
            "Win": "--",
        }
        write_csv(tables_dir / "tableA_sana06b_bds_by_split.csv", split_rows, [])
        write_markdown_table(tables_dir / "tableA_sana06b_bds_by_split.md", ["No completed gains"], [["Need more data"]])
        write_csv(tables_dir / "cross_model_bds_row.csv", [cross_row])
        write_markdown_table(tables_dir / "cross_model_bds_row.md", list(cross_row.keys()), [list(cross_row.values())])
        write_csv(tables_dir / "table_cross_model_same_compute_sana06b_row.csv", [cross_row])
        write_markdown_table(
            tables_dir / "table_cross_model_same_compute_sana06b_row.md",
            list(cross_row.keys()),
            [list(cross_row.values())],
        )
        (tables_dir / "table_cross_model_same_compute_sana06b_row.tex").write_text(
            "Sana-0.6B & Image Generation & None / not explicit & 50 & 0 & Need more data & -- & -- & -- & -- & -- & -- \\\\\n",
            encoding="utf-8",
        )
        (reports_dir / "03_bds_report.md").write_text(
            "# BDS report\n\nNo completed same-compute BSS/uniform metric pairs were available. Verdict: Need more data.\n",
            encoding="utf-8",
        )
        print("No gains found; wrote Need more data BDS placeholders.")
        return

    case_ids = sorted({row["case_id"] for row in gains})
    by_case_t = {(row["case_id"], int(row["actual_nfe"])): float(row["gain"]) for row in gains}
    closures = {
        (row["case_id"], int(row["actual_nfe"])): float(row["rgb_l1_closure"])
        for row in master
        if row.get("method_family") == "bss"
    }
    available_t = sorted({int(row["actual_nfe"]) for row in gains})
    all_t = [t for t in [10, 20, 30, 40] if t in available_t]
    low_t = [10] if 10 in available_t else []

    split_rows = []
    low_lcbs = []
    all_lcbs = []
    for split_name, (cal, holdout) in split_cases(case_ids).items():
        for tset_name, tset in [("BDS_low", low_t), ("BDS_all", all_t)]:
            cal_values = [by_case_t[(case, t)] for case in cal for t in tset if (case, t) in by_case_t]
            holdout_values = [by_case_t[(case, t)] for case in holdout for t in tset if (case, t) in by_case_t]
            lcb, lo, hi = bootstrap_ci(cal_values, args.bootstrap_iters, seed=0)
            if tset_name == "BDS_low":
                low_lcbs.append(lcb)
            else:
                all_lcbs.append(lcb)
            split_rows.append(
                {
                    "split": split_name,
                    "tset": tset_name,
                    "tset_values": ",".join(str(t) for t in tset),
                    "calibration_cases": len(cal),
                    "holdout_cases": len(holdout),
                    "calibration_mean_gain": f"{mean(cal_values):.8f}" if cal_values else "",
                    "bootstrap_lcb95": f"{lcb:.8f}" if cal_values else "",
                    "bootstrap_ci95_low": f"{lo:.8f}" if cal_values else "",
                    "bootstrap_ci95_high": f"{hi:.8f}" if cal_values else "",
                    "calibration_win_rate": f"{mean([1.0 if v > 0 else 0.0 for v in cal_values]):.6f}" if cal_values else "",
                    "holdout_mean_gain": f"{mean(holdout_values):.8f}" if holdout_values else "",
                    "holdout_win_rate": f"{mean([1.0 if v > 0 else 0.0 for v in holdout_values]):.6f}" if holdout_values else "",
                }
            )

    overall_low_lcb = min(low_lcbs) if low_lcbs else 0.0
    overall_all_lcb = min(all_lcbs) if all_lcbs else 0.0
    final_verdict = verdict(overall_low_lcb, overall_all_lcb, len(case_ids))
    cells = {}
    gains_by_t = {}
    for t in [10, 20, 30, 40]:
        t_gains = [by_case_t[(case, t)] for case in case_ids if (case, t) in by_case_t]
        t_closures = [closures[(case, t)] for case in case_ids if (case, t) in closures]
        gains_by_t[t] = t_gains
        cells[t] = format_cell(mean(t_closures) if t_closures else None, mean(t_gains) if t_gains else None, t)
    all_gain_values = [value for values in gains_by_t.values() for value in values]
    mean_delta = f"{mean(all_gain_values):+.3f}" if all_gain_values else "--"
    win = f"{mean([1.0 if v > 0 else 0.0 for v in all_gain_values]):.3f}" if all_gain_values else "--"
    cross_row = {
        "Model": "Sana-0.6B",
        "Setting": "Image Generation",
        "Few-step prior": "None / not explicit",
        "Ref.": "50",
        "Cases": str(len(case_ids)),
        "BDS": final_verdict,
        "Low": cells[10],
        "Middle Low": cells[20],
        "Middle High": cells[30],
        "High": cells[40],
        "Mean Delta": mean_delta,
        "Win": win,
    }

    write_csv(tables_dir / "tableA_sana06b_bds_by_split.csv", split_rows)
    write_markdown_table(
        tables_dir / "tableA_sana06b_bds_by_split.md",
        list(split_rows[0].keys()) if split_rows else ["No completed gains"],
        [list(r.values()) for r in split_rows] if split_rows else [["Need more data"]],
    )
    for name in ["cross_model_bds_row", "table_cross_model_same_compute_sana06b_row"]:
        write_csv(tables_dir / f"{name}.csv", [cross_row])
        write_markdown_table(tables_dir / f"{name}.md", list(cross_row.keys()), [list(cross_row.values())])
    (tables_dir / "table_cross_model_same_compute_sana06b_row.tex").write_text(
        " & ".join(cross_row.values()) + " \\\\\n",
        encoding="utf-8",
    )
    (reports_dir / "03_bds_report.md").write_text(
        "# BDS report\n\n"
        f"Cases: {len(case_ids)}\n\n"
        f"Available T values: {', '.join(str(t) for t in available_t)}\n\n"
        f"Verdict: {final_verdict}\n\n"
        f"Conservative low LCB95: {overall_low_lcb:.6f}\n\n"
        f"Conservative all LCB95: {overall_all_lcb:.6f}\n",
        encoding="utf-8",
    )
    print(f"Wrote BDS tables to {tables_dir}; verdict={final_verdict}")


if __name__ == "__main__":
    main()
