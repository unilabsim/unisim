"""Check the pinned Newton runtime boundary without importing engine SDKs.

The default probe only inspects installed distribution metadata and module
availability. Passing ``--import`` performs the optional imports explicitly;
this may initialize CUDA-facing runtime code and is therefore not part of the
base test path.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
from importlib import metadata

PINNED_DISTRIBUTIONS: dict[str, str] = {
    "newton": "1.5.1",
    "mujoco-warp": "3.11.0",
    "mujoco": "3.11.0",
    "warp-lang": "1.16.0",
}
MODULES: dict[str, str] = {
    "newton": "newton",
    "mujoco-warp": "mujoco_warp",
    "mujoco": "mujoco",
    "warp-lang": "warp",
}


def check_runtime(*, import_modules: bool = False) -> list[str]:
    """Return human-readable diagnostics; an empty list means the probe passed."""
    diagnostics: list[str] = []
    if sys.version_info < (3, 10):
        diagnostics.append(
            f"Python >= 3.10 is required, found {sys.version_info.major}.{sys.version_info.minor}"
        )

    for distribution, expected in PINNED_DISTRIBUTIONS.items():
        try:
            found = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            diagnostics.append(f"{distribution} is not installed (expected {expected})")
            continue
        if found != expected:
            diagnostics.append(f"{distribution}=={found} is installed; expected {expected}")
        module_name = MODULES[distribution]
        if importlib.util.find_spec(module_name) is None:
            diagnostics.append(f"module {module_name!r} is unavailable for {distribution}")
            continue
        if import_modules:
            try:
                importlib.import_module(module_name)
            except Exception as exc:  # pragma: no cover - depends on native runtime
                diagnostics.append(f"import {module_name!r} failed: {type(exc).__name__}: {exc}")
    return diagnostics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--import",
        dest="import_modules",
        action="store_true",
        help="import the optional Newton stack after metadata checks",
    )
    args = parser.parse_args()
    diagnostics = check_runtime(import_modules=args.import_modules)
    if diagnostics:
        print("Newton runtime probe: FAILED")
        for diagnostic in diagnostics:
            print(f"- {diagnostic}")
        print("Install the isolated stack with: uv sync --extra newton")
        print("Do not combine it with: uv sync --extra mjwarp")
        return 1
    print("Newton runtime probe: OK")
    for distribution, version in PINNED_DISTRIBUTIONS.items():
        print(f"- {distribution}=={version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
