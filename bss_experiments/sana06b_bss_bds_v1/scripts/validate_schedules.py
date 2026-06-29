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
    build_bss_flow_schedule,
    official_time_uniform_flow_schedule,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    checks = {}
    for nfe in [8, 10, 20, 30, 40, 50]:
        coords = official_time_uniform_flow_schedule(nfe, flow_shift=DEFAULT_FLOW_SHIFT)
        checks[f"uniform{nfe}"] = {
            "evals": len(coords) - 1,
            "first": coords[0],
            "last": coords[-1],
            "pass": len(coords) - 1 == nfe,
        }
    for nfe in [10, 20, 30, 40]:
        base, bss, inserted = build_bss_flow_schedule(nfe, flow_shift=DEFAULT_FLOW_SHIFT)
        uniform = official_time_uniform_flow_schedule(nfe, flow_shift=DEFAULT_FLOW_SHIFT)
        checks[f"bss{nfe}"] = {
            "evals": len(bss) - 1,
            "base_evals": len(base) - 1,
            "inserted_count": len(inserted),
            "differs_from_uniform": any(abs(a - b) > 1e-12 for a, b in zip(bss, uniform)),
            "pass": len(bss) - 1 == nfe and len(base) - 1 == nfe - 2 and len(inserted) == 2,
        }
    failed = [name for name, payload in checks.items() if not payload["pass"]]
    text = json.dumps({"checks": checks, "failed": failed}, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    print(text)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
