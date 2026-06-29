from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (  # noqa: E402
    DEFAULT_FLOW_SHIFT,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_HEIGHT,
    DEFAULT_PAG_GUIDANCE_SCALE,
    DEFAULT_REFERENCE_NFE,
    DEFAULT_WIDTH,
    EXPERIMENT_ID,
    MODEL_ID,
    MODEL_VARIANT,
    ensure_dir,
    experiment_dir,
    find_repo_root,
    git_info,
    prompt_hash,
    write_csv,
)


SMOKE_PROMPT = 'a cyberpunk cat with a neon sign that says "Sana"'

PROMPTS = [
    ("p01_simple_object", "simple object", "A matte red ceramic teapot on a plain gray table, studio lighting."),
    ("p02_animal", "animal", "A golden retriever wearing a blue raincoat, sitting beside a puddle after rain."),
    ("p03_human_action", "human action", "A chef tossing vegetables in a wok inside a busy open kitchen."),
    ("p04_indoor_scene", "indoor scene", "A quiet reading room with oak shelves, a green lamp, and a leather chair."),
    ("p05_outdoor_landscape", "outdoor landscape", "A wide alpine valley at sunrise with snow peaks and a winding river."),
    ("p06_high_frequency_texture", "high-frequency texture", "A close view of woven tartan fabric with crisp red and green threads."),
    ("p07_lighting_heavy", "lighting-heavy scene", "A glass greenhouse at dusk filled with warm string lights and mist."),
    ("p08_text_sign", "text/sign prompt", 'A storefront sign that clearly says "SANA LAB" above a small flower shop.'),
    ("p09_surreal_fantasy", "surreal fantasy", "A floating island with tiny waterfalls under two moons, painterly but sharp."),
    ("p10_architecture", "architecture", "A modern concrete library with tall arches, reflected in polished black stone."),
    ("p11_food_product", "food/product", "A glossy product photo of a lemon tart on a white plate with mint garnish."),
    ("p12_macro", "close-up macro", "A macro photograph of a dew-covered blue butterfly wing with fine detail."),
    ("p13_reflection", "reflection/water/glass", "A sailboat reflected in still water beside a transparent glass pier."),
    ("p14_complex_composition", "complex composition", "A market square with cyclists, umbrellas, fruit stands, and distant tram lines."),
    ("p15_low_light", "low-light scene", "A low-light jazz club with a saxophonist under a narrow amber spotlight."),
    ("p16_colorful_abstract", "colorful abstract scene", "An abstract burst of cyan, magenta, yellow, and black ink in clear water."),
]


FIELDNAMES = [
    "run_id",
    "model_id",
    "model_variant",
    "setting",
    "few_step_prior",
    "task",
    "modality",
    "protocol",
    "backend",
    "case_id",
    "category",
    "prompt",
    "prompt_hash",
    "method",
    "method_family",
    "sampler_mode",
    "actual_nfe",
    "num_inference_steps",
    "base_sample_steps",
    "split_pairs",
    "scheduler_name",
    "flow_shift",
    "guidance_scale",
    "pag_guidance_scale",
    "seed",
    "height",
    "width",
    "reference_method",
    "reference_nfe",
    "low_baseline_method",
    "output_path",
    "schedule_json_path",
    "stdout_log_path",
    "stderr_log_path",
    "runtime_json_path",
    "status",
    "error_message",
    "git_commit",
    "dirty_status",
]


def method_spec(method: str) -> dict[str, str | int]:
    if method == "reference_uniform50":
        return {
            "method_family": "reference",
            "sampler_mode": "uniform",
            "actual_nfe": 50,
            "num_inference_steps": 50,
            "base_sample_steps": "",
            "split_pairs": "",
        }
    if method.startswith("uniform"):
        nfe = int(method.replace("uniform", ""))
        return {
            "method_family": "uniform",
            "sampler_mode": "uniform",
            "actual_nfe": nfe,
            "num_inference_steps": nfe,
            "base_sample_steps": "",
            "split_pairs": "",
        }
    if method.startswith("bss"):
        nfe = int(method.replace("bss", ""))
        return {
            "method_family": "bss",
            "sampler_mode": "bss",
            "actual_nfe": nfe,
            "num_inference_steps": nfe,
            "base_sample_steps": nfe - 2,
            "split_pairs": "0,-1",
        }
    raise ValueError(f"Unknown method {method}")


def build_row(
    *,
    split: str,
    case_id: str,
    category: str,
    prompt: str,
    method: str,
    backend: str,
    seed: int,
    height: int,
    width: int,
    guidance_scale: float,
    pag_guidance_scale: float,
    git: dict[str, str],
) -> dict:
    spec = method_spec(method)
    stem = f"{case_id}_{method}"
    return {
        "run_id": f"{EXPERIMENT_ID}_{split}_{stem}",
        "model_id": MODEL_ID,
        "model_variant": MODEL_VARIANT,
        "setting": "Image Generation",
        "few_step_prior": "None / not explicit",
        "task": "t2i",
        "modality": "image",
        "protocol": "fixed_prompt_suite_official_sana_pipeline",
        "backend": backend,
        "case_id": case_id,
        "category": category,
        "prompt": prompt,
        "prompt_hash": prompt_hash(prompt),
        "method": method,
        "method_family": spec["method_family"],
        "sampler_mode": spec["sampler_mode"],
        "actual_nfe": spec["actual_nfe"],
        "num_inference_steps": spec["num_inference_steps"],
        "base_sample_steps": spec["base_sample_steps"],
        "split_pairs": spec["split_pairs"],
        "scheduler_name": "flow_dpm-solver",
        "flow_shift": DEFAULT_FLOW_SHIFT,
        "guidance_scale": guidance_scale,
        "pag_guidance_scale": pag_guidance_scale,
        "seed": seed,
        "height": height,
        "width": width,
        "reference_method": "reference_uniform50",
        "reference_nfe": DEFAULT_REFERENCE_NFE,
        "low_baseline_method": "uniform8",
        "output_path": f"images/{split}/{stem}.png",
        "schedule_json_path": f"schedules/{split}/{stem}.json",
        "stdout_log_path": f"logs/{split}/{stem}.stdout.log",
        "stderr_log_path": f"logs/{split}/{stem}.stderr.log",
        "runtime_json_path": f"runtimes/{split}/{stem}.runtime.json",
        "status": "pending",
        "error_message": "",
        "git_commit": git["commit"],
        "dirty_status": git["dirty_status"],
    }


def write_prompt_suite(path: Path) -> None:
    payload = {
        "suite_id": "sana06b_prompt_suite_v1",
        "model_id": MODEL_ID,
        "model_variant": MODEL_VARIANT,
        "purpose": "T2I cross-modality BSS/BDS sanity check, not an official benchmark",
        "selection_rule": "Fixed before model outputs are inspected; do not cherry-pick.",
        "smoke": {
            "case_id": "smoke_official",
            "category": "official smoke",
            "prompt": SMOKE_PROMPT,
            "seed": 42,
        },
        "prompts": [
            {"index": idx, "case_id": case_id, "category": category, "prompt": prompt, "seed": 0}
            for idx, (case_id, category, prompt) in enumerate(PROMPTS, start=1)
        ],
    }
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", default=None)
    parser.add_argument("--backend", default="native", choices=["native", "diffusers"])
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--guidance-scale", type=float, default=DEFAULT_GUIDANCE_SCALE)
    parser.add_argument("--pag-guidance-scale", type=float, default=DEFAULT_PAG_GUIDANCE_SCALE)
    args = parser.parse_args()

    repo_root = find_repo_root()
    out_root = Path(args.experiment_root).resolve() if args.experiment_root else experiment_dir(repo_root)
    git = git_info(repo_root)
    ensure_dir(out_root / "manifests")
    ensure_dir(out_root / "prompt_suites")

    prompt_suite_path = out_root / "prompt_suites" / "sana06b_prompt_suite_v1.json"
    write_prompt_suite(prompt_suite_path)

    smoke_rows = [
        build_row(
            split="smoke",
            case_id="smoke_official",
            category="official smoke",
            prompt=SMOKE_PROMPT,
            method=method,
            backend=args.backend,
            seed=42,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
            pag_guidance_scale=args.pag_guidance_scale,
            git=git,
        )
        for method in ["uniform8", "uniform10", "bss10", "reference_uniform50"]
    ]
    write_csv(out_root / "manifests" / "sana06b_smoke_manifest.csv", smoke_rows, FIELDNAMES)

    methods = [
        "uniform8",
        "uniform10",
        "uniform20",
        "uniform30",
        "uniform40",
        "reference_uniform50",
        "bss10",
        "bss20",
        "bss30",
        "bss40",
    ]
    suite_rows = []
    for case_id, category, prompt in PROMPTS:
        for method in methods:
            suite_rows.append(
                build_row(
                    split="mini",
                    case_id=case_id,
                    category=category,
                    prompt=prompt,
                    method=method,
                    backend=args.backend,
                    seed=0,
                    height=args.height,
                    width=args.width,
                    guidance_scale=args.guidance_scale,
                    pag_guidance_scale=args.pag_guidance_scale,
                    git=git,
                )
            )
    write_csv(out_root / "manifests" / "sana06b_prompt_suite_manifest.csv", suite_rows, FIELDNAMES)
    print(f"Wrote manifests under {out_root / 'manifests'}")
    print(f"Wrote prompt suite to {prompt_suite_path}")


if __name__ == "__main__":
    main()
