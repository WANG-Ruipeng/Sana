from __future__ import annotations

import argparse
import importlib
import os
import platform
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    HF_DIFFUSERS_REPO_ID,
    HF_PTH_REPO_ID,
    default_weights_dir,
    ensure_dir,
    experiment_dir,
    find_repo_root,
    git_info,
    run_cmd,
    write_markdown_table,
)


def check_import(module_name: str, attr: str | None = None) -> tuple[bool, str]:
    try:
        module = importlib.import_module(module_name)
        if attr:
            getattr(module, attr)
        version = getattr(module, "__version__", "")
        return True, version or "import OK"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def find_weight_paths(weights_root: Path) -> list[Path]:
    candidates = [
        weights_root / "checkpoints" / "Sana_600M_1024px.pth",
        weights_root / "checkpoint" / "Sana_600M_1024px.pth",
        weights_root / "Sana_600M_1024px.pth",
    ]
    found = [path for path in candidates if path.exists()]
    if weights_root.exists():
        found.extend(path for path in weights_root.rglob("*.pth") if path not in found)
    return found


def row(item: str, found: bool | str, evidence: str, notes: str, action: str) -> list[str]:
    found_text = found if isinstance(found, str) else ("yes" if found else "no")
    return [item, found_text, evidence.replace("\n", "<br>"), notes, action]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-dir", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    repo_root = find_repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    exp_root = experiment_dir(repo_root)
    out_path = Path(args.output) if args.output else exp_root / "reports" / "00_repo_model_hardware_audit.md"
    ensure_dir(out_path.parent)
    git = git_info(repo_root)

    torch_ok, torch_version = check_import("torch")
    cuda_version = "torch unavailable"
    cuda_available = False
    gpu_name = "none"
    gpu_vram = "unknown"
    if torch_ok:
        import torch

        cuda_available = torch.cuda.is_available()
        cuda_version = torch.version.cuda or "not built with CUDA"
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            gpu_vram = f"{props.total_memory / (1024**3):.1f} GiB"

    nvidia_rc, nvidia_out, nvidia_err = run_cmd(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], repo_root, timeout=10
    )
    nvidia_evidence = nvidia_out if nvidia_rc == 0 else nvidia_err or "nvidia-smi unavailable"

    diffusers_ok, diffusers_version = check_import("diffusers")
    diffusers_sana_ok, diffusers_sana_evidence = check_import("diffusers", "SanaPipeline")
    native_ok, native_evidence = check_import("app.sana_pipeline", "SanaPipeline")

    weights_dir = Path(args.weights_dir or os.environ.get("SANA06B_WEIGHTS_DIR") or default_weights_dir())
    found_weights = find_weight_paths(weights_dir)

    in_colab = "google.colab" in sys.modules or Path("/content").exists()
    drive_mounted = Path("/content/drive/MyDrive").exists()
    app_pipeline = repo_root / "app" / "sana_pipeline.py"
    config_600m = repo_root / DEFAULT_CONFIG_PATH
    config_1600m = repo_root / "configs/sana_config/1024ms/Sana_1600M_img1024.yaml"

    rows = [
        row("current repo path", True, str(repo_root), "", "none"),
        row("git remote", bool(git["remote"]), git["remote"], "", "none"),
        row("current branch", True, git["branch"], "", "none"),
        row("commit hash", True, git["commit"], "", "none"),
        row("dirty status", git["dirty_status"] == "clean", git["dirty_status"], "", "review before commit"),
        row("Python version", True, platform.python_version(), sys.executable, "none"),
        row("PyTorch version", torch_ok, torch_version, "", "install torch in Colab if missing"),
        row("CUDA version", cuda_available, cuda_version, "", "use Colab GPU runtime for experiment"),
        row("nvidia-smi", nvidia_rc == 0, nvidia_evidence, "", "use GPU runtime if unavailable"),
        row("GPU name and VRAM", cuda_available, f"{gpu_name}; {gpu_vram}", "", "use >=16GB if possible"),
        row("running in Colab", in_colab, str(in_colab), "", "intended runner is Colab"),
        row("Drive mounted", drive_mounted, "/content/drive/MyDrive", "", "mount Drive in notebook"),
        row("Sana official repo exists locally", True, str(repo_root), "current checkout is a Sana repo/fork", "none"),
        row("app/sana_pipeline.py exists", app_pipeline.exists(), str(app_pipeline), "", "none"),
        row("Sana_600M_img1024.yaml exists", config_600m.exists(), str(config_600m), "", "none"),
        row("Sana_1600M_img1024.yaml exists", config_1600m.exists(), str(config_1600m), "", "none"),
        row("diffusers import", diffusers_ok, diffusers_version, "", "pip install diffusers>=0.32.0 if missing"),
        row("diffusers SanaPipeline usable", diffusers_sana_ok, diffusers_sana_evidence, "", "native remains primary"),
        row("diffusers>=0.32.0", diffusers_ok, diffusers_version, "", "upgrade if version is below 0.32.0"),
        row("native SanaPipeline import", native_ok, native_evidence, "from app.sana_pipeline import SanaPipeline", "install repo deps if missing"),
        row("HF pth model repo id", True, HF_PTH_REPO_ID, "primary native checkpoint candidate", "audit model card before run"),
        row("HF diffusers model repo id", True, HF_DIFFUSERS_REPO_ID, "fallback candidate listed in local docs", "do not force if unavailable"),
        row(
            "Drive weights for Sana_600M_1024px",
            bool(found_weights),
            "<br>".join(str(p) for p in found_weights) if found_weights else str(weights_dir),
            "weights must stay outside git",
            "download to Drive if missing",
        ),
    ]

    download_cmd = (
        f"huggingface-cli download {HF_PTH_REPO_ID} "
        f'--local-dir "$DRIVE_WEIGHTS_ROOT/Sana_600M_1024px"'
    )
    notes = [
        "# Repo/model/hardware audit",
        "",
        "This audit is intentionally local and non-training. It does not download weights.",
        "",
        "Important model mapping note: local Sana docs list both the pth repo "
        f"`{HF_PTH_REPO_ID}` and diffusers repo `{HF_DIFFUSERS_REPO_ID}` for Sana-0.6B 1024px. "
        f"The selected native config is `{DEFAULT_CONFIG_PATH}`, whose model string is `SanaMS_600M_P1_D28` "
        "and flow shift is 4.0. The runner still validates the checkpoint path at load time.",
        "",
    ]
    out_path.write_text("\n".join(notes), encoding="utf-8")
    write_markdown_table(out_path, ["Item", "Found?", "Path / Evidence", "Notes", "Action"], rows)
    with out_path.open("a", encoding="utf-8") as fh:
        fh.write("\n## Model mapping note\n\n")
        fh.write(
            "Local Sana docs list both the pth repo "
            f"`{HF_PTH_REPO_ID}` and diffusers repo `{HF_DIFFUSERS_REPO_ID}` for Sana-0.6B 1024px. "
            f"The selected native config is `{DEFAULT_CONFIG_PATH}`, whose model string is `SanaMS_600M_P1_D28` "
            "and flow shift is 4.0. The runner still validates the checkpoint path at load time.\n"
        )
        if not found_weights:
            fh.write("## Missing weights action\n\n")
            fh.write("Weights were not found locally. Download them to Google Drive, not this repo:\n\n")
            fh.write(f"```bash\n{download_cmd}\n```\n")
        fh.write("\n## Backend selection\n\n")
        fh.write(
            "Backend A (`native`) is selected as primary for this experiment because it uses "
            "`app.sana_pipeline.SanaPipeline` with the local 600M config and the official Flow-DPM-Solver path. "
            "Backend B (`diffusers`) is available only if the 0.6B diffusers checkpoint loads and exposes compatible "
            "timestep control for BSS.\n"
        )
    print(out_path)


if __name__ == "__main__":
    main()
