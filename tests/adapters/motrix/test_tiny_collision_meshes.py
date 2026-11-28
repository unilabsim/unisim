"""Colliding meshes below MotrixSim's convex-cook volume threshold keep a collider."""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest

pytest.importorskip("motrixsim")

import motrixsim.msd as msd  # noqa: E402

from unisim.backend.motrix.scene import _prepare_imported_world  # noqa: E402

# A 4 mm x 3 mm x 0.5 mm closed box: far below the engine's 2**-23 m^3 threshold.
HALF_EXTENTS = (0.002, 0.0015, 0.00025)
BOX_FACES = (
    (1, 3, 4),
    (1, 4, 2),
    (5, 7, 8),
    (5, 8, 6),
    (1, 5, 6),
    (1, 6, 2),
    (3, 7, 8),
    (3, 8, 4),
    (1, 5, 7),
    (1, 7, 3),
    (2, 6, 8),
    (2, 8, 4),
)


def _write_scene(tmp_path: Path, scale: float = 1.0) -> Path:
    corners = itertools.product((-1, 1), repeat=3)
    (tmp_path / "tiny.obj").write_text(
        "".join(
            f"v {x * HALF_EXTENTS[0] * scale:.6f} {y * HALF_EXTENTS[1] * scale:.6f}"
            f" {z * HALF_EXTENTS[2] * scale:.6f}\n"
            for x, y, z in corners
        )
        + "".join(f"f {a} {b} {c}\n" for a, b, c in BOX_FACES),
        encoding="utf-8",
    )
    scene = tmp_path / "scene.xml"
    scene.write_text(
        """<mujoco model='tiny-collision-mesh'>
  <worldbody>
    <body name='base' pos='0 0 .05'>
      <freejoint name='root'/>
      <inertial pos='0 0 0' mass='.1' diaginertia='1e-5 1e-5 1e-5'/>
      <geom name='tiny' type='mesh' mesh='tiny' density='0'/>
    </body>
  </worldbody>
  <asset><mesh name='tiny' file='tiny.obj'/></asset>
</mujoco>""",
        encoding="utf-8",
    )
    return scene


def _collider_count(world) -> int:
    return len(msd.build(world).low.colliders)


def test_tiny_collision_mesh_keeps_a_collider(tmp_path: Path) -> None:
    scene = _write_scene(tmp_path)
    assert _collider_count(msd.from_file(str(scene))) == 0
    world = msd.from_file(str(scene))
    _prepare_imported_world(world, scene)
    geom = world.find_body("base").link.geoms[0]
    assert geom.collision_cooking == msd.MeshCollisionCookPolicy.BoundingBox
    assert _collider_count(world) == 1


def test_regular_collision_mesh_keeps_its_convex_cook(tmp_path: Path) -> None:
    scene = _write_scene(tmp_path, scale=20.0)
    world = msd.from_file(str(scene))
    _prepare_imported_world(world, scene)
    geom = world.find_body("base").link.geoms[0]
    assert geom.collision_cooking is None
    assert _collider_count(world) == 1
