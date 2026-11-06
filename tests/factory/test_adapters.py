import pytest

import unisim
from unisim.optional import OptionalDependencyError


def test_manifest_covers_all_current_backends():
    names = {spec.name for spec in unisim.ADAPTER_SPECS}
    assert names == {
        "mujoco",
        "motrix",
        "drake",
        "mjwarp",
        "newton",
        "superdex",
        "genesis",
        "isaacgym",
        "isaacsim",
    }
    assert all(spec.status == "available" for spec in unisim.ADAPTER_SPECS)


@pytest.mark.parametrize(
    ("backend_type", "export_name"),
    [
        ("mujoco", "MuJoCoBackend"),
        ("motrix", "MotrixBackend"),
        ("drake", "DrakeBackend"),
        ("mjwarp", "MjwarpBackend"),
        ("newton", "NewtonBackend"),
        ("superdex", "SuperDexBackend"),
        ("genesis", "GenesisBackend"),
        ("isaacgym", "IsaacGymBackend"),
        ("isaacsim", "IsaacSimBackend"),
    ],
)
def test_every_manifest_adapter_has_a_lazy_public_class(backend_type: str, export_name: str):
    spec = unisim.adapter_spec(backend_type)
    adapter = getattr(unisim, export_name)

    assert spec.status == "available"
    assert isinstance(adapter, type)
    assert adapter.__module__.startswith("unisim.backend.")


@pytest.mark.parametrize(
    "backend_type", ["mujoco", "motrix", "drake", "mjwarp", "newton", "genesis", "superdex"]
)
def test_in_process_adapters_require_scene(backend_type):
    with pytest.raises(ValueError, match="requires a SceneCfg"):
        unisim.create_backend(backend_type)


@pytest.mark.parametrize("backend_type", ["isaacgym", "isaacsim"])
def test_worker_adapters_fail_closed_without_runtime(backend_type, monkeypatch):
    monkeypatch.delenv("UNISIM_ISAACGYM_PYTHON", raising=False)
    monkeypatch.delenv("UNISIM_ISAACSIM_PYTHON", raising=False)
    with pytest.raises(OptionalDependencyError):
        unisim.create_backend(backend_type)


def test_factory_drake_warns_and_ignores_mujoco_only_options(tmp_path, monkeypatch):
    from unisim.backend.drake import backend as drake_backend
    from unisim.scene import SceneCfg

    calls: list[dict[str, object]] = []

    class _RoutedBackend:
        def __init__(self, *args: object, **kwargs: object) -> None:
            calls.append({"args": args, "kwargs": kwargs})

    # Stubbing the adapter class keeps this factory test free of the optional
    # Drake SDK; the constructor never runs the real DrakeUni loader.
    monkeypatch.setattr(drake_backend, "DrakeBackend", _RoutedBackend)
    scene = SceneCfg(model_file=str(tmp_path / "drake.xml"))
    with pytest.warns(
        UserWarning,
        match=r"drake ignores non-default MuJoCo-only backend options: "
        r"post_step_forward_sensor=True, iterations=4, chunk_size=8, cpu_ids=\[0, 1\], "
        r"bench_nsteps=3",
    ):
        unisim.create_backend(
            "drake",
            scene=scene,
            num_envs=2,
            sim_dt=0.002,
            post_step_forward_sensor=True,
            iterations=4,
            chunk_size=8,
            cpu_ids=[0, 1],
            bench_nsteps=3,
        )

    assert len(calls) == 1
    kwargs = calls[0]["kwargs"]
    assert kwargs == {"drake_backend_mode": "batch"}


@pytest.mark.parametrize(
    ("option", "expected"),
    [(None, None), (True, True), (False, False)],
)
def test_factory_routes_mjwarp_validate_tensor_finite(tmp_path, monkeypatch, option, expected):
    from unisim.backend.mjwarp import backend as mjwarp_backend
    from unisim.scene import SceneCfg

    calls: list[dict[str, object]] = []

    class _RoutedBackend:
        def __init__(self, *args: object, **kwargs: object) -> None:
            calls.append(kwargs)

    # Stubbing the adapter class keeps this routing test free of CUDA/Warp.
    monkeypatch.setattr(mjwarp_backend, "MjwarpBackend", _RoutedBackend)
    unisim.create_backend(
        "mjwarp",
        scene=SceneCfg(model_file=str(tmp_path / "mjwarp.xml")),
        num_envs=2,
        sim_dt=0.002,
        mjwarp_validate_tensor_finite=option,
    )

    assert len(calls) == 1
    assert "mjwarp_validate_tensor_finite" not in calls[0]
    assert calls[0].get("validate_tensor_finite") is expected


def test_factory_rejects_non_bool_mjwarp_validate_tensor_finite(tmp_path):
    from unisim.scene import SceneCfg

    with pytest.raises(TypeError, match="mjwarp_validate_tensor_finite must be bool"):
        unisim.create_backend(
            "mjwarp",
            scene=SceneCfg(model_file=str(tmp_path / "mjwarp.xml")),
            num_envs=2,
            sim_dt=0.002,
            mjwarp_validate_tensor_finite=1,
        )


def test_factory_drops_mjwarp_validate_tensor_finite_for_other_backends(tmp_path, monkeypatch):
    from unisim.backend.drake import backend as drake_backend
    from unisim.scene import SceneCfg

    calls: list[dict[str, object]] = []

    class _RoutedBackend:
        def __init__(self, *args: object, **kwargs: object) -> None:
            calls.append(kwargs)

    monkeypatch.setattr(drake_backend, "DrakeBackend", _RoutedBackend)
    unisim.create_backend(
        "drake",
        scene=SceneCfg(model_file=str(tmp_path / "drake.xml")),
        num_envs=2,
        sim_dt=0.002,
        mjwarp_validate_tensor_finite=False,
    )

    assert calls == [{"drake_backend_mode": "batch"}]



def test_protocol_round_trip():
    from unisim.backend.subprocess_ipc.protocol import decode_message, pack_message

    assert decode_message(pack_message("PING", {"value": 1})) == {
        "cmd": "PING",
        "payload": {"value": 1},
    }
