"""Benchmark API reservation.

No workload runner or measurement implementation is provided in v0.1.x. These
records define the stable extension point future benchmark work must consume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    """Identifies a future reproducible physics workload."""

    name: str
    schema_version: str = "0.1"
    scene_digest: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Machine-readable result envelope reserved for future runners."""

    case: BenchmarkCase
    backend: str
    package_version: str
    metrics: Mapping[str, float] = field(default_factory=dict)
    provenance: Mapping[str, str] = field(default_factory=dict)
