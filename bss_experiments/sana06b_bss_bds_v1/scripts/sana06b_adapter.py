from __future__ import annotations

import contextlib
import inspect
import os
import sys
import time
import types
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    DEFAULT_FLOW_SHIFT,
    DEFAULT_PAG_GUIDANCE_SCALE,
    HF_DIFFUSERS_REPO_ID,
    HF_PTH_REPO_ID,
    build_bss_flow_schedule,
    git_info,
    official_time_uniform_flow_schedule,
    resolve_artifact_path,
    schedule_payload,
    validate_schedule_payload,
    write_json,
)


def _install_mmcv_registry_fallback() -> None:
    try:
        import mmcv  # noqa: F401
        from mmcv.runner import get_dist_info as _get_dist_info  # noqa: F401
        from mmcv.utils.logging import logger_initialized as _logger_initialized  # noqa: F401
        return
    except (ImportError, ModuleNotFoundError):
        for module_name in ["mmcv", "mmcv.utils", "mmcv.utils.logging", "mmcv.runner"]:
            sys.modules.pop(module_name, None)

    mmcv_mod = types.ModuleType("mmcv")
    mmcv_mod.__path__ = []

    class Config(dict):
        def __init__(self, cfg_dict=None, **kwargs):
            super().__init__()
            data = {}
            if cfg_dict:
                data.update(dict(cfg_dict))
            data.update(kwargs)
            for key, value in data.items():
                self[key] = self._wrap(value)

        @staticmethod
        def _wrap(value):
            if isinstance(value, dict) and not isinstance(value, Config):
                return Config(value)
            if isinstance(value, list):
                return [Config._wrap(item) for item in value]
            return value

        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

        def __setattr__(self, name, value):
            self[name] = self._wrap(value)

        def merge_from_dict(self, options):
            for key, value in dict(options).items():
                self[key] = self._wrap(value)

        @classmethod
        def fromfile(cls, filename):
            import yaml

            with open(filename, encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
            return cls(data)

    class Registry:
        def __init__(self, name, *args, **kwargs):
            self.name = name
            self.module_dict = {}
            self._module_dict = self.module_dict

        def __contains__(self, key):
            return key in self.module_dict

        def get(self, key):
            return self.module_dict.get(key)

        def register_module(self, module=None, name=None, force=False):
            def _register(cls):
                module_name = name or cls.__name__
                if not force and module_name in self.module_dict:
                    raise KeyError(f"{module_name} is already registered in {self.name}")
                self.module_dict[module_name] = cls
                return cls

            if module is not None:
                return _register(module)
            return _register

        def build(self, cfg, default_args=None):
            return build_from_cfg(cfg, self, default_args=default_args)

    def build_from_cfg(cfg, registry, default_args=None):
        if cfg is None:
            raise TypeError("cfg must not be None")
        if isinstance(cfg, str):
            cfg = {"type": cfg}
        elif hasattr(cfg, "to_dict"):
            cfg = cfg.to_dict()
        else:
            cfg = dict(cfg)
        args = dict(default_args or {})
        args.update({k: v for k, v in cfg.items() if k != "type"})
        obj_type = cfg.get("type")
        if isinstance(obj_type, str):
            obj_cls = registry.get(obj_type)
            if obj_cls is None:
                raise KeyError(f"{obj_type} is not registered in {registry.name}")
        else:
            obj_cls = obj_type
        return obj_cls(**args)

    def mkdir_or_exist(dir_name):
        Path(dir_name).mkdir(parents=True, exist_ok=True)

    def dump(obj, file):
        import pickle

        with open(file, "wb") as handle:
            pickle.dump(obj, handle)

    def load(file):
        import pickle

        with open(file, "rb") as handle:
            return pickle.load(handle)

    mmcv_mod.Config = Config
    mmcv_mod.Registry = Registry
    mmcv_mod.build_from_cfg = build_from_cfg
    mmcv_mod.mkdir_or_exist = mkdir_or_exist
    mmcv_mod.dump = dump
    mmcv_mod.load = load

    utils_mod = types.ModuleType("mmcv.utils")
    try:
        from torch.nn.modules.batchnorm import _BatchNorm
        from torch.nn.modules.instancenorm import _InstanceNorm
    except Exception:
        _BatchNorm = tuple()
        _InstanceNorm = tuple()
    utils_mod._BatchNorm = _BatchNorm
    utils_mod._InstanceNorm = _InstanceNorm

    logging_mod = types.ModuleType("mmcv.utils.logging")
    logging_mod.logger_initialized = {}
    utils_mod.logging = logging_mod

    runner_mod = types.ModuleType("mmcv.runner")
    runner_mod.OPTIMIZERS = Registry("optimizer")
    runner_mod.OPTIMIZER_BUILDERS = Registry("optimizer builder")

    class DefaultOptimizerConstructor:
        def __init__(self, optimizer_cfg, paramwise_cfg=None):
            self.optimizer_cfg = dict(optimizer_cfg or {})
            self.paramwise_cfg = paramwise_cfg or {}
            self.base_lr = self.optimizer_cfg.get("lr")
            self.base_wd = self.optimizer_cfg.get("weight_decay")

        def __call__(self, model):
            return build_optimizer(model, self.optimizer_cfg)

        def add_params(self, params, module, prefix="", is_dcn_module=None):
            params.extend({"params": [param], "name": name} for name, param in module.named_parameters(recurse=False))

        @staticmethod
        def _is_in(param_group, params):
            target = set(param_group.get("params", []))
            return any(bool(target.intersection(set(group.get("params", [])))) for group in params)

    def get_dist_info():
        try:
            import torch

            if torch.distributed.is_available() and torch.distributed.is_initialized():
                return torch.distributed.get_rank(), torch.distributed.get_world_size()
        except Exception:
            pass
        return 0, 1

    def build_optimizer(model, optimizer_cfg):
        import torch

        cfg = dict(optimizer_cfg or {})
        opt_type = cfg.pop("type", "AdamW")
        opt_cls = getattr(torch.optim, opt_type) if isinstance(opt_type, str) else opt_type
        return opt_cls(model.parameters(), **cfg)

    runner_mod.DefaultOptimizerConstructor = DefaultOptimizerConstructor
    runner_mod.get_dist_info = get_dist_info
    runner_mod.build_optimizer = build_optimizer
    mmcv_mod.utils = utils_mod
    mmcv_mod.runner = runner_mod

    sys.modules["mmcv"] = mmcv_mod
    sys.modules["mmcv.utils"] = utils_mod
    sys.modules["mmcv.utils.logging"] = logging_mod
    sys.modules["mmcv.runner"] = runner_mod



class Sana06BAdapter:
    def __init__(
        self,
        backend: str,
        repo_root: str | Path,
        weights_dir: str | Path | None,
        config_path: str | Path | None,
        device: str = "cuda",
        dtype: str = "float16",
        experiment_root: str | Path | None = None,
    ):
        self.backend = backend
        self.repo_root = Path(repo_root).resolve()
        self.weights_dir = Path(weights_dir).expanduser() if weights_dir else None
        self.config_path = Path(config_path or DEFAULT_CONFIG_PATH)
        if not self.config_path.is_absolute():
            self.config_path = self.repo_root / self.config_path
        self.device = device
        self.dtype = dtype
        self.experiment_root = Path(experiment_root).resolve() if experiment_root else self.repo_root
        self.pipeline = None

    def load_pipeline(self):
        if self.pipeline is not None:
            return self.pipeline
        if self.backend == "native":
            self.pipeline = self._load_native_pipeline()
        elif self.backend == "diffusers":
            self.pipeline = self._load_diffusers_pipeline()
        else:
            raise ValueError(f"Unsupported backend {self.backend}; expected native or diffusers")
        return self.pipeline

    def get_official_base_schedule(self, num_inference_steps: int, **kwargs) -> list[float]:
        return official_time_uniform_flow_schedule(
            num_inference_steps,
            flow_shift=float(kwargs.get("flow_shift", DEFAULT_FLOW_SHIFT)),
        )

    def build_uniform_schedule(self, actual_nfe: int, **kwargs) -> dict:
        flow_shift = float(kwargs.get("flow_shift", DEFAULT_FLOW_SHIFT))
        coords = official_time_uniform_flow_schedule(actual_nfe, flow_shift=flow_shift)
        return {
            "base_coords": coords,
            "final_coords": coords,
            "inserted_midpoints": [],
            "coordinate_type": "native_flow_dpm_solver_time_uniform_flow_shifted_sigma",
        }

    def build_bss_schedule(self, actual_nfe: int, split_pairs=(0, -1), **kwargs) -> dict:
        flow_shift = float(kwargs.get("flow_shift", DEFAULT_FLOW_SHIFT))
        base_coords, final_coords, inserted = build_bss_flow_schedule(
            actual_nfe,
            split_pairs=tuple(split_pairs),
            flow_shift=flow_shift,
        )
        return {
            "base_coords": base_coords,
            "final_coords": final_coords,
            "inserted_midpoints": inserted,
            "coordinate_type": "native_flow_dpm_solver_time_uniform_flow_shifted_sigma",
        }

    def run_one_case(self, manifest_row: dict[str, str]) -> dict[str, Any]:
        pipeline = self.load_pipeline()
        sampler_mode = manifest_row["sampler_mode"]
        actual_nfe = int(manifest_row["actual_nfe"])
        split_pairs = self._parse_split_pairs(manifest_row.get("split_pairs", "0,-1"))
        flow_shift = float(manifest_row.get("flow_shift") or DEFAULT_FLOW_SHIFT)
        if sampler_mode == "bss":
            schedule = self.build_bss_schedule(actual_nfe, split_pairs=split_pairs, flow_shift=flow_shift)
        else:
            schedule = self.build_uniform_schedule(actual_nfe, flow_shift=flow_shift)

        payload = schedule_payload(
            manifest_row=manifest_row,
            sampler_mode=sampler_mode,
            final_coords=schedule["final_coords"],
            base_coords=schedule["base_coords"],
            inserted_midpoints=schedule["inserted_midpoints"],
            git=git_info(self.repo_root),
        )
        errors = validate_schedule_payload(payload)
        if errors:
            raise ValueError("; ".join(errors))
        self.dump_schedule_json(manifest_row, payload)

        output_path = resolve_artifact_path(manifest_row["output_path"], self.experiment_root)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        runtime_path = resolve_artifact_path(manifest_row["runtime_json_path"], self.experiment_root)
        runtime_path.parent.mkdir(parents=True, exist_ok=True)

        start = time.perf_counter()
        if self.backend == "native":
            result = self._run_native(pipeline, manifest_row, schedule if sampler_mode == "bss" else None)
        else:
            result = self._run_diffusers(pipeline, manifest_row, schedule if sampler_mode == "bss" else None)
        elapsed = time.perf_counter() - start
        self._save_image_result(result, output_path)

        runtime_payload = {
            "run_id": manifest_row["run_id"],
            "case_id": manifest_row["case_id"],
            "method": manifest_row["method"],
            "backend": self.backend,
            "actual_nfe": actual_nfe,
            "seconds": elapsed,
            "output_path": str(output_path),
            "schedule_json_path": str(resolve_artifact_path(manifest_row["schedule_json_path"], self.experiment_root)),
        }
        write_json(runtime_path, runtime_payload)
        self.validate_outputs(manifest_row)
        return runtime_payload

    def dump_schedule_json(self, manifest_row: dict[str, str], schedule_payload_: dict) -> None:
        path = resolve_artifact_path(manifest_row["schedule_json_path"], self.experiment_root)
        write_json(path, schedule_payload_)

    def validate_outputs(self, manifest_row: dict[str, str]) -> None:
        output_path = resolve_artifact_path(manifest_row["output_path"], self.experiment_root)
        schedule_path = resolve_artifact_path(manifest_row["schedule_json_path"], self.experiment_root)
        if not output_path.exists():
            raise FileNotFoundError(f"Expected output image missing: {output_path}")
        if output_path.stat().st_size <= 0:
            raise RuntimeError(f"Output image is empty: {output_path}")
        if not schedule_path.exists():
            raise FileNotFoundError(f"Expected schedule JSON missing: {schedule_path}")

    def _load_native_pipeline(self):
        if str(self.repo_root) not in sys.path:
            sys.path.insert(0, str(self.repo_root))
        _install_mmcv_registry_fallback()
        from app.sana_pipeline import SanaPipeline

        checkpoint = self._resolve_native_checkpoint()
        pipe = SanaPipeline(str(self.config_path))
        pipe.from_pretrained(str(checkpoint))
        return pipe

    def _load_diffusers_pipeline(self):
        import torch
        from diffusers import SanaPipeline

        dtype = self._torch_dtype(torch)
        model_source = str(self.weights_dir) if self.weights_dir and self.weights_dir.exists() else HF_DIFFUSERS_REPO_ID
        pipe = SanaPipeline.from_pretrained(model_source, variant="fp16", torch_dtype=dtype)
        pipe.to(self.device)
        if hasattr(pipe, "vae"):
            pipe.vae.to(torch.bfloat16)
        if hasattr(pipe, "text_encoder"):
            pipe.text_encoder.to(torch.bfloat16)
        return pipe

    def _run_native(self, pipeline, row: dict[str, str], schedule: dict | None):
        import torch

        device = torch.device(self.device if torch.cuda.is_available() or self.device == "cpu" else "cpu")
        generator = torch.Generator(device=device).manual_seed(int(row["seed"]))
        kwargs = {
            "prompt": row["prompt"],
            "height": int(row["height"]),
            "width": int(row["width"]),
            "guidance_scale": float(row["guidance_scale"]),
            "pag_guidance_scale": float(row.get("pag_guidance_scale") or DEFAULT_PAG_GUIDANCE_SCALE),
            "num_inference_steps": int(row["actual_nfe"]),
            "generator": generator,
        }
        if schedule is None:
            return pipeline(**kwargs)
        with self._patched_native_flow_schedule(schedule["final_coords"]):
            return pipeline(**kwargs)

    def _run_diffusers(self, pipeline, row: dict[str, str], schedule: dict | None):
        import torch

        if schedule is not None:
            signature = inspect.signature(pipeline.__call__)
            if "timesteps" not in signature.parameters:
                raise NotImplementedError(
                    "Diffusers SanaPipeline did not expose a timesteps argument in this environment; "
                    "use backend=native for BSS schedule injection."
                )
        generator = torch.Generator(device=self.device if self.device != "cpu" else "cpu").manual_seed(int(row["seed"]))
        kwargs = {
            "prompt": row["prompt"],
            "height": int(row["height"]),
            "width": int(row["width"]),
            "guidance_scale": float(row["guidance_scale"]),
            "num_inference_steps": int(row["actual_nfe"]),
            "generator": generator,
        }
        if schedule is not None:
            kwargs["timesteps"] = schedule["final_coords"]
        return pipeline(**kwargs)

    @contextlib.contextmanager
    def _patched_native_flow_schedule(self, coords: list[float]):
        import torch
        import diffusion.model.dpm_solver as dpm_solver_mod

        original = dpm_solver_mod.DPM_Solver.get_time_steps

        def patched_get_time_steps(solver, skip_type, t_T, t_0, N, device, shift=1.0):
            if skip_type == "time_uniform_flow" and int(N) == len(coords) - 1:
                return torch.tensor(coords, dtype=torch.float32, device=device)
            return original(solver, skip_type, t_T, t_0, N, device, shift=shift)

        dpm_solver_mod.DPM_Solver.get_time_steps = patched_get_time_steps
        try:
            yield
        finally:
            dpm_solver_mod.DPM_Solver.get_time_steps = original

    def _save_image_result(self, result, output_path: Path) -> None:
        if hasattr(result, "images"):
            image = result.images[0]
            image.save(output_path)
            return
        if isinstance(result, (list, tuple)) and result and hasattr(result[0], "save"):
            result[0].save(output_path)
            return
        import torch
        from torchvision.utils import save_image

        tensor = result[0] if isinstance(result, (list, tuple)) else result
        if not torch.is_tensor(tensor):
            raise TypeError(f"Unsupported pipeline output type: {type(result)!r}")
        save_image(tensor, output_path, nrow=1, normalize=True, value_range=(-1, 1))

    def _resolve_native_checkpoint(self) -> Path | str:
        if self.weights_dir is None:
            raise FileNotFoundError(self._weights_missing_message())
        raw = str(self.weights_dir)
        if raw.startswith("hf://"):
            return raw
        root = Path(os.path.expanduser(raw))
        candidates = []
        if root.is_file():
            candidates.append(root)
        else:
            candidates.extend(
                [
                    root / "checkpoints" / "Sana_600M_1024px.pth",
                    root / "checkpoint" / "Sana_600M_1024px.pth",
                    root / "Sana_600M_1024px.pth",
                ]
            )
            if root.exists():
                candidates.extend(sorted(root.rglob("*.pth")))
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(self._weights_missing_message())

    def _weights_missing_message(self) -> str:
        target = self.weights_dir or "<DRIVE_WEIGHTS_ROOT>/Sana_600M_1024px"
        return (
            f"Sana-0.6B checkpoint not found at {target}. Download to Drive, not the repo, for example:\n"
            f"huggingface-cli download {HF_PTH_REPO_ID} --local-dir \"$DRIVE_WEIGHTS_ROOT/Sana_600M_1024px\""
        )

    def _torch_dtype(self, torch_module):
        if self.dtype in {"float16", "fp16"}:
            return torch_module.float16
        if self.dtype in {"bfloat16", "bf16"}:
            return torch_module.bfloat16
        if self.dtype in {"float32", "fp32"}:
            return torch_module.float32
        raise ValueError(f"Unsupported dtype {self.dtype}")

    @staticmethod
    def _parse_split_pairs(raw: str) -> tuple[int, int]:
        if not raw:
            return (0, -1)
        values = [int(part.strip()) for part in str(raw).split(",") if part.strip()]
        if len(values) != 2:
            raise ValueError(f"split_pairs must contain two comma-separated ints, got {raw!r}")
        return (values[0], values[1])
