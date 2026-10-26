"""Authored MJCF collision filtering on the real CPU Newton runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from unisim.backend.newton.materialization import (
    build_source_builder,
    scan_newton_model_metadata,
)
from unisim.dr.types import ModelSourceDescriptor
from unisim.entities import SceneEntitySpec
from unisim.mjcf_compiler import compose_scene
from unisim.scene import SceneCfg

mujoco = pytest.importorskip("mujoco")
newton = pytest.importorskip("newton")


def _write_scene(tmp_path: Path, worldbody: str, contact: str = "") -> Path:
    path = tmp_path / "contact_filtering.xml"
    path.write_text(
        "<mujoco model='contact-filtering'>"
        "<option gravity='0 0 0'/>"
        f"<worldbody>{worldbody}</worldbody>{contact}</mujoco>",
        encoding="utf-8",
    )
    return path


def _free_sphere(name: str, x: float, z: float = 0.5, attributes: str = "") -> str:
    return (
        f"<body name='{name}' pos='{x} 0 {z}'><freejoint name='{name}_root'/>"
        f"<geom name='{name}_geom' type='sphere' size='.1' mass='1' {attributes}/>"
        "</body>"
    )


def _source_solver(path: Path) -> tuple[Any, Any]:
    metadata = scan_newton_model_metadata(mujoco, SceneCfg(model_file=str(path)))
    builder = build_source_builder(newton, str(path), metadata=metadata)
    model = builder.finalize(device="cpu")
    solver = newton.solvers.SolverMuJoCo(model, nconmax=32, njmax=32)
    return metadata, solver


def _compiled_geom_names(solver: Any) -> list[str]:
    return [
        str(
            mujoco.mj_id2name(
                solver.mj_model,
                mujoco.mjtObj.mjOBJ_GEOM,
                geom_id,
            )
            or ""
        ).rsplit("/", 1)[-1]
        for geom_id in range(int(solver.mj_model.ngeom))
    ]


def _matches_geom_name(compiled: str, authored: str) -> bool:
    base, separator, suffix = compiled.rpartition("_")
    return compiled == authored or (bool(separator) and base == authored and suffix.isdigit())


def _geom_shape(solver: Any, metadata: Any, name: str) -> int:
    authored_id = mujoco.mj_name2id(
        metadata.playback_model,
        mujoco.mjtObj.mjOBJ_GEOM,
        name,
    )
    assert authored_id >= 0
    compiled_id = next(
        geom_id
        for geom_id, compiled in enumerate(_compiled_geom_names(solver))
        if _matches_geom_name(compiled, name)
    )
    mapping = np.asarray(solver.mjc_geom_to_newton_shape.numpy(), dtype=np.int64)
    return int(mapping.reshape(mapping.shape[0], -1)[0, compiled_id])


def _contact_pairs(solver: Any, contacts: Any) -> set[frozenset[int]]:
    solver.update_contacts(contacts)
    count = int(np.asarray(contacts.rigid_contact_count.numpy()).reshape(-1)[0])
    shape0 = np.asarray(contacts.rigid_contact_shape0.numpy()[:count], dtype=np.int64)
    shape1 = np.asarray(contacts.rigid_contact_shape1.numpy()[:count], dtype=np.int64)
    return {frozenset((int(a), int(b))) for a, b in zip(shape0, shape1, strict=True)}


def _step_source_contacts(solver: Any) -> set[frozenset[int]]:
    model = solver.model
    state = model.state()
    state_out = model.state()
    control = model.control()
    contacts = newton.Contacts(solver.get_max_contact_count(), 0, device="cpu")
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    state.clear_forces()
    solver.step(state, state_out, control, contacts, 0.001)
    return _contact_pairs(solver, contacts)


def test_contact_exclude_filters_only_the_authored_pair(tmp_path: Path) -> None:
    path = _write_scene(
        tmp_path,
        _free_sphere("excluded_a", 0.0)
        + _free_sphere("excluded_b", 0.0, 0.65)
        + _free_sphere("included_a", 1.0)
        + _free_sphere("included_b", 1.0, 0.65),
        "<contact><exclude body1='excluded_a' body2='excluded_b'/></contact>",
    )
    metadata, solver = _source_solver(path)
    excluded = frozenset(
        (
            _geom_shape(solver, metadata, "excluded_a_geom"),
            _geom_shape(solver, metadata, "excluded_b_geom"),
        )
    )
    included = frozenset(
        (
            _geom_shape(solver, metadata, "included_a_geom"),
            _geom_shape(solver, metadata, "included_b_geom"),
        )
    )

    pairs = _step_source_contacts(solver)
    assert pairs == {included}
    assert excluded not in pairs


def test_asymmetric_masks_and_visual_only_geoms_keep_mujoco_semantics(
    tmp_path: Path,
) -> None:
    path = _write_scene(
        tmp_path,
        _free_sphere("one_way_a", 0.0, attributes="contype='1' conaffinity='0'")
        + _free_sphere("one_way_b", 0.0, 0.65, "contype='0' conaffinity='1'")
        + _free_sphere("blocked_a", 1.0, attributes="contype='1' conaffinity='0'")
        + _free_sphere("blocked_b", 1.0, 0.65, "contype='2' conaffinity='0'")
        + _free_sphere("visual", 2.0, attributes="contype='0' conaffinity='0'")
        + _free_sphere("solid", 2.0, 0.65),
    )
    metadata, solver = _source_solver(path)
    one_way = frozenset(
        (
            _geom_shape(solver, metadata, "one_way_a_geom"),
            _geom_shape(solver, metadata, "one_way_b_geom"),
        )
    )
    blocked = frozenset(
        (
            _geom_shape(solver, metadata, "blocked_a_geom"),
            _geom_shape(solver, metadata, "blocked_b_geom"),
        )
    )
    solid = _geom_shape(solver, metadata, "solid_geom")
    assert not any(
        _matches_geom_name(compiled, "visual_geom")
        for compiled in _compiled_geom_names(solver)
    )

    pairs = _step_source_contacts(solver)
    assert pairs == {one_way}
    assert blocked not in pairs
    assert all(solid not in pair for pair in pairs)


def test_authored_plane_collides_without_a_duplicate_dynamic_floor(tmp_path: Path) -> None:
    path = _write_scene(
        tmp_path,
        "<geom name='ground' type='plane' size='2 2 .1'/>"
        + _free_sphere("ball", 0.0, 0.08),
    )
    metadata, solver = _source_solver(path)
    model = solver.model
    shape_type = np.asarray(model.shape_type.numpy(), dtype=np.int64)
    shape_body = np.asarray(model.shape_body.numpy(), dtype=np.int64)
    plane = int(newton.GeoType.PLANE)
    assert int(model.shape_count) == 2
    assert np.count_nonzero(shape_type == plane) == 1
    assert int(shape_body[np.flatnonzero(shape_type == plane)[0]]) == -1

    ground = _geom_shape(solver, metadata, "ground")
    ball = _geom_shape(solver, metadata, "ball_geom")
    assert _step_source_contacts(solver) == {frozenset((ground, ball))}


def test_scene_contact_excludes_reach_newton_after_entity_composition(tmp_path: Path) -> None:
    specs = []
    for name in ("a", "b", "c"):
        source = tmp_path / f"{name}.xml"
        source.write_text(
            f"<mujoco model='{name}'><worldbody>{_free_sphere(name, 0.0)}</worldbody></mujoco>",
            encoding="utf-8",
        )
        specs.append(SceneEntitySpec(name, ModelSourceDescriptor(str(source))))
    scene = SceneCfg(entity_assets=tuple(specs), contact_excludes=(("a/a", "b/b"),))
    with compose_scene(scene, num_envs=1, sim_dt=0.001) as composed:
        _metadata, solver = _source_solver(Path(composed.model_file))

    shapes = {}
    mapping = np.asarray(solver.mjc_geom_to_newton_shape.numpy(), dtype=np.int64)
    mapping = mapping.reshape(mapping.shape[0], -1)[0]
    for geom_id, compiled in enumerate(_compiled_geom_names(solver)):
        for name in ("a", "b", "c"):
            if _matches_geom_name(compiled, f"{name}_geom"):
                shapes[name] = int(mapping[geom_id])
    assert set(shapes) == {"a", "b", "c"}
    pairs = _step_source_contacts(solver)
    assert pairs == {
        frozenset((shapes["a"], shapes["c"])),
        frozenset((shapes["b"], shapes["c"])),
    }
    assert frozenset((shapes["a"], shapes["b"])) not in pairs
