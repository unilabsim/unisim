"""Platform topology selection for SuperDex's default native worker count."""

from unittest.mock import patch

from unisim.backend.superdex import cpu_topology


def test_macos_physical_cpu_count_uses_sysctl() -> None:
    with (
        patch.object(cpu_topology, "_available_cpu_ids", return_value=list(range(8))),
        patch.object(cpu_topology.platform, "system", return_value="Darwin"),
        patch.object(cpu_topology.subprocess, "check_output", return_value="4\n"),
    ):
        assert cpu_topology.physical_cpu_count() == 4
        assert cpu_topology.physical_cpu_groups() == [[0, 1], [2, 3], [4, 5], [6, 7]]


def test_linux_affinity_counts_one_worker_per_core() -> None:
    reads = {"cpu0": ("0", "0"), "cpu1": ("0", "1"), "cpu2": ("0", "0"), "cpu3": ("0", "1")}

    def read_text(path):
        cpu = next(name for name in reads if name in str(path))
        return reads[cpu][0 if str(path).endswith("physical_package_id") else 1]

    with (
        patch.object(cpu_topology, "_available_cpu_ids", return_value=[0, 1, 2, 3]),
        patch.object(cpu_topology.platform, "system", return_value="Linux"),
        patch.object(cpu_topology.Path, "read_text", autospec=True, side_effect=read_text),
    ):
        assert cpu_topology.physical_cpu_groups() == [[0, 2], [1, 3]]
