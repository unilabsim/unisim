"""Portable physical CPU discovery for the SuperDex native worker pool."""

from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path


def _available_cpu_ids() -> list[int]:
    try:
        return sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return list(range(os.cpu_count() or 1))


def _linux_physical_groups(cpu_ids: list[int]) -> list[list[int]]:
    groups: dict[tuple[str, str], list[int]] = {}
    topology = Path("/sys/devices/system/cpu")
    for cpu_id in cpu_ids:
        try:
            package = (topology / f"cpu{cpu_id}/topology/physical_package_id").read_text().strip()
            core = (topology / f"cpu{cpu_id}/topology/core_id").read_text().strip()
        except OSError:
            return []
        groups.setdefault((package, core), []).append(cpu_id)
    return list(groups.values())


def physical_cpu_groups() -> list[list[int]]:
    """Return affinity-visible logical CPUs grouped by physical core.

    Linux exposes an exact package/core mapping. macOS does not expose a
    stable logical-CPU-to-core mapping, so its physical count is used to form
    deterministic contiguous groups; this keeps the worker count correct and
    avoids relying on Linux-only affinity APIs.
    """
    cpu_ids = _available_cpu_ids()
    if platform.system() == "Linux":
        groups = _linux_physical_groups(cpu_ids)
        if groups:
            return groups
    if platform.system() == "Darwin":
        try:
            physical = int(
                subprocess.check_output(
                    ["sysctl", "-n", "hw.physicalcpu"], text=True, timeout=1
                ).strip()
            )
            physical = max(1, min(physical, len(cpu_ids)))
            groups = [[] for _ in range(physical)]
            for index, cpu_id in enumerate(cpu_ids):
                groups[min(index * physical // max(1, len(cpu_ids)), physical - 1)].append(cpu_id)
            return [group for group in groups if group]
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    return [[cpu_id] for cpu_id in cpu_ids]


def physical_cpu_count() -> int:
    """Return the number of physical cores visible to this process."""
    return max(1, len(physical_cpu_groups()))
