"""Contract tests for debug overlay primitives and the typed camera config."""

from __future__ import annotations

import dataclasses
import textwrap
from pathlib import Path

import numpy as np
import pytest

from unisim.backend.base import (
    BackendPlayCapabilities,
    CameraCfg,
    DebugPrimitive,
    validate_debug_overlays,
)
from unisim.fake import FakeBackend


class TestDebugPrimitive:
    def test_sphere_defaults(self) -> None:
        primitive = DebugPrimitive(kind="sphere", pos=(0.0, 0.0, 0.5), size=(0.025,))
        assert primitive.rgba == (1.0, 0.2, 0.2, 0.5)
        assert primitive.quat is None
        assert primitive.mesh_asset is None
        assert primitive.text is None

    def test_frozen(self) -> None:
        primitive = DebugPrimitive(kind="sphere", pos=(0, 0, 0), size=(0.1,))
        with pytest.raises(dataclasses.FrozenInstanceError):
            primitive.pos = (1, 1, 1)  # type: ignore[misc]

    def test_unknown_kind_fails(self) -> None:
        with pytest.raises(ValueError, match="kind must be one of"):
            DebugPrimitive(kind="cone", pos=(0, 0, 0))  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("kind", "size"),
        [
            ("sphere", (0.1, 0.2)),
            ("box", (0.1,)),
            ("frame", ()),
            ("arrow", (0.1, 0.2)),
            ("ghost_geom", (1.0, 2.0)),
            ("text", (1.0,)),
        ],
    )
    def test_size_arity_enforced(self, kind: str, size: tuple[float, ...]) -> None:
        kwargs: dict = {"kind": kind, "pos": (0, 0, 0), "size": size}
        if kind == "ghost_geom":
            kwargs["mesh_asset"] = "asset.stl"
        if kind == "text":
            kwargs["text"] = "label"
        with pytest.raises(ValueError, match="size arity"):
            DebugPrimitive(**kwargs)

    def test_non_positive_size_fails(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            DebugPrimitive(kind="sphere", pos=(0, 0, 0), size=(-0.1,))

    def test_non_finite_pos_fails(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            DebugPrimitive(kind="sphere", pos=(0, float("nan"), 0), size=(0.1,))

    def test_quat_must_be_unit_wxyz(self) -> None:
        with pytest.raises(ValueError, match="unit-length"):
            DebugPrimitive(
                kind="box", pos=(0, 0, 0), size=(0.1, 0.1, 0.1), quat=(2.0, 0.0, 0.0, 0.0)
            )
        with pytest.raises(ValueError, match="non-zero"):
            DebugPrimitive(
                kind="box", pos=(0, 0, 0), size=(0.1, 0.1, 0.1), quat=(0.0, 0.0, 0.0, 0.0)
            )

    def test_rgba_range_enforced(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            DebugPrimitive(kind="sphere", pos=(0, 0, 0), size=(0.1,), rgba=(1.0, 0, 0, 1.5))

    def test_ghost_geom_requires_mesh_asset(self) -> None:
        with pytest.raises(ValueError, match="mesh_asset"):
            DebugPrimitive(kind="ghost_geom", pos=(0, 0, 0))
        primitive = DebugPrimitive(kind="ghost_geom", pos=(0, 0, 0), mesh_asset="goal.stl")
        assert primitive.size == ()

    def test_mesh_asset_rejected_for_other_kinds(self) -> None:
        with pytest.raises(ValueError, match="only valid for kind 'ghost_geom'"):
            DebugPrimitive(kind="sphere", pos=(0, 0, 0), size=(0.1,), mesh_asset="x.stl")

    def test_text_rules(self) -> None:
        with pytest.raises(ValueError, match="requires a text string"):
            DebugPrimitive(kind="text", pos=(0, 0, 0))
        with pytest.raises(ValueError, match="only valid for kind 'text'"):
            DebugPrimitive(kind="sphere", pos=(0, 0, 0), size=(0.1,), text="hi")
        primitive = DebugPrimitive(kind="text", pos=(0, 0, 0), text="goal")
        assert primitive.text == "goal"


class TestValidateDebugOverlays:
    def test_none_passes_through(self) -> None:
        assert validate_debug_overlays(None, 2) is None

    def test_valid_overlay_shape(self) -> None:
        overlays = [
            [DebugPrimitive(kind="sphere", pos=(0, 0, 0), size=(0.1,))],
            None,
        ]
        assert validate_debug_overlays(overlays, 2) is overlays

    def test_outer_length_must_match_num_envs(self) -> None:
        with pytest.raises(ValueError, match=r"len == 3"):
            validate_debug_overlays([[]], 3)

    def test_non_sequence_outer_fails(self) -> None:
        with pytest.raises(TypeError, match="per-env sequence"):
            validate_debug_overlays("nope", 1)  # type: ignore[arg-type]

    def test_non_primitive_entry_fails(self) -> None:
        with pytest.raises(TypeError, match="DebugPrimitive"):
            validate_debug_overlays([[{"kind": "sphere"}]], 1)


class TestCameraCfg:
    def test_defaults(self) -> None:
        cfg = CameraCfg.from_kwargs(None)
        assert cfg.cam_distance == 2.0
        assert cfg.cam_elevation == -20.0
        assert cfg.cam_azimuth == 90.0
        assert cfg.cam_lookat is None
        assert not cfg.cam_tracking
        assert cfg.cam_fov is None

    def test_passthrough(self) -> None:
        cfg = CameraCfg(cam_distance=3.0, cam_tracking=True, cam_tracking_env_idx=2)
        assert CameraCfg.from_kwargs(cfg) is cfg

    def test_mapping_subset(self) -> None:
        cfg = CameraCfg.from_kwargs(
            {"cam_distance": 1.5, "cam_lookat": [0, 0, 0.5], "cam_fov": 45.0}
        )
        assert cfg.cam_distance == 1.5
        assert cfg.cam_lookat == (0.0, 0.0, 0.5)
        assert cfg.cam_fov == 45.0

    def test_unknown_keys_fail_closed_with_names(self) -> None:
        with pytest.raises(ValueError, match="unknown camera_kwargs key.*'cam_zoom'"):
            CameraCfg.from_kwargs({"cam_distance": 2.0, "cam_zoom": 3})

    @pytest.mark.parametrize("legacy_key", ["distance", "elevation_deg", "azimuth_deg"])
    def test_legacy_alias_keys_fail_closed(self, legacy_key: str) -> None:
        with pytest.raises(ValueError, match=f"'{legacy_key}'"):
            CameraCfg.from_kwargs({legacy_key: 1.0})

    def test_bad_lookat_fails(self) -> None:
        with pytest.raises(ValueError, match="cam_lookat must have 3 components"):
            CameraCfg.from_kwargs({"cam_lookat": [0, 0]})

    def test_bad_values_fail(self) -> None:
        with pytest.raises(ValueError, match="cam_distance"):
            CameraCfg(cam_distance=0.0)
        with pytest.raises(ValueError, match="cam_elevation"):
            CameraCfg(cam_elevation=-120.0)
        with pytest.raises(ValueError, match="cam_fov"):
            CameraCfg(cam_fov=200.0)
        with pytest.raises(ValueError, match="cam_tracking_env_idx"):
            CameraCfg(cam_tracking_env_idx=-1)

    def test_non_mapping_fails(self) -> None:
        with pytest.raises(TypeError, match="CameraCfg, a mapping, or None"):
            CameraCfg.from_kwargs([("cam_distance", 2.0)])  # type: ignore[arg-type]


class TestPlayCapabilities:
    def test_default_is_false(self) -> None:
        assert not BackendPlayCapabilities().supports_debug_overlay
        assert not BackendPlayCapabilities().supports_interactive_debug_overlay

    def test_fake_backend_default(self) -> None:
        capabilities = FakeBackend().get_play_capabilities()
        assert not capabilities.supports_debug_overlay
        assert not capabilities.supports_interactive_debug_overlay

    @pytest.mark.parametrize(
        "backend_path",
        [
            "unisim.backend.drake.backend.DrakeBackend",
            "unisim.backend.mjwarp.backend.MjwarpBackend",
            "unisim.backend.newton.backend.NewtonBackend",
            "unisim.backend.superdex.backend.SuperDexBackend",
        ],
    )
    def test_mujoco_snapshot_pipeline_backends_support_overlay(self, backend_path: str) -> None:
        module_path, _, class_name = backend_path.rpartition(".")
        import importlib

        backend_cls = getattr(importlib.import_module(module_path), class_name)
        backend = backend_cls.__new__(backend_cls)
        assert backend.get_play_capabilities().supports_debug_overlay

    def test_only_mjwarp_supports_interactive_overlay(self) -> None:
        import importlib

        for module_path, class_name, expected in (
            ("unisim.backend.mjwarp.backend", "MjwarpBackend", True),
            ("unisim.backend.drake.backend", "DrakeBackend", False),
            ("unisim.backend.newton.backend", "NewtonBackend", False),
            ("unisim.backend.superdex.backend", "SuperDexBackend", False),
            ("unisim.backend.motrix.backend", "MotrixBackend", False),
            ("unisim.backend.genesis.backend", "GenesisBackend", False),
            ("unisim.backend.subprocess_ipc.backend", "MjcfSubprocessBackend", False),
        ):
            backend_cls = getattr(importlib.import_module(module_path), class_name)
            backend = backend_cls.__new__(backend_cls)
            assert (
                backend.get_play_capabilities().supports_interactive_debug_overlay is expected
            ), f"{class_name} interactive overlay capability mismatch"

    def test_motrix_genesis_subprocess_do_not_support_overlay(self) -> None:
        import importlib

        for module_path, class_name in (
            ("unisim.backend.motrix.backend", "MotrixBackend"),
            ("unisim.backend.genesis.backend", "GenesisBackend"),
            ("unisim.backend.subprocess_ipc.backend", "MjcfSubprocessBackend"),
        ):
            backend_cls = getattr(importlib.import_module(module_path), class_name)
            backend = backend_cls.__new__(backend_cls)
            assert not backend.get_play_capabilities().supports_debug_overlay


class TestRunPlaybackContract:
    def test_base_run_playback_fails_closed(self) -> None:
        with pytest.raises(NotImplementedError, match="does not support playback execution"):
            FakeBackend().run_playback(env=None, initialize=lambda: None, step=lambda o: o,
                                       num_steps=1)

    @pytest.mark.parametrize(
        ("module_path", "class_name"),
        [
            ("unisim.backend.motrix.backend", "MotrixBackend"),
            ("unisim.backend.genesis.backend", "GenesisBackend"),
            ("unisim.backend.subprocess_ipc.backend", "MjcfSubprocessBackend"),
        ],
    )
    def test_unsupported_backends_fail_closed_on_overlay(
        self, module_path: str, class_name: str
    ) -> None:
        import importlib

        backend_cls = getattr(importlib.import_module(module_path), class_name)
        backend = backend_cls.__new__(backend_cls)
        with pytest.raises(NotImplementedError, match="debug overlay primitives"):
            backend.run_playback(
                env=None,
                initialize=lambda: None,
                step=lambda o: o,
                num_steps=1,
                debug_overlay_getter=lambda: None,
            )

    def test_mjwarp_interactive_accepts_overlay_getter(self, monkeypatch) -> None:
        """An overlay getter no longer fails closed; the run now reaches the
        platform guards/model loading instead of the removed NotImplementedError."""
        from unisim.backend.mjwarp.playback import run_mjwarp_playback

        class _SentinelBackend:
            def get_playback_model(self, world: int) -> str:
                raise RuntimeError("SENTINEL reached model loading")

        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.delenv("MUJOCO_GL", raising=False)
        with pytest.raises(RuntimeError, match="DISPLAY|SENTINEL"):
            run_mjwarp_playback(
                backend=_SentinelBackend(),
                env=None,
                initialize=lambda: None,
                step=lambda o: o,
                num_steps=1,
                output_video=None,
                render_spacing=None,
                headless=False,
                record_video=False,
                snapshot_shape=(1, 2),
                frame_state_getter=None,
                camera_kwargs=None,
                debug_overlay_getter=lambda: None,
            )


class TestPrimitiveToGeomConversion:
    """Primitive -> mjvGeom conversion without a GL context (MjvScene only)."""

    @pytest.fixture(autouse=True)
    def _mujoco(self):
        self.mujoco = pytest.importorskip("mujoco")
        from unisim.visualization import render_many

        self.render_many = render_many
        self.model = self.mujoco.MjModel.from_xml_string(
            "<mujoco><worldbody><geom type='box' size='0.05 0.05 0.05'/></worldbody></mujoco>"
        )

    def _scene(self, maxgeom: int = 32):
        return self.mujoco.MjvScene(self.model, maxgeom=maxgeom)

    def test_sphere_box_arrow_conversion(self) -> None:
        scene = self._scene()
        overlays = [
            [
                DebugPrimitive(kind="sphere", pos=(0, 0, 0.5), size=(0.05,)),
                DebugPrimitive(
                    kind="box",
                    pos=(0.1, 0, 0.5),
                    size=(0.02, 0.03, 0.04),
                    quat=(1.0, 0.0, 0.0, 0.0),
                    rgba=(0.0, 1.0, 0.0, 0.4),
                ),
                DebugPrimitive(kind="arrow", pos=(0, 0, 0.5), size=(0.2,)),
            ]
        ]
        self.render_many.append_debug_primitives(
            scene, overlays, offsets=None, mesh_ids={}
        )
        assert scene.ngeom == 3
        sphere, box, arrow = scene.geoms[0], scene.geoms[1], scene.geoms[2]
        assert int(sphere.type) == int(self.mujoco.mjtGeom.mjGEOM_SPHERE)
        np.testing.assert_allclose(sphere.pos, [0, 0, 0.5])
        assert sphere.size[0] == pytest.approx(0.05)
        assert int(box.type) == int(self.mujoco.mjtGeom.mjGEOM_BOX)
        np.testing.assert_allclose(box.size, [0.02, 0.03, 0.04], atol=1e-7)
        np.testing.assert_allclose(box.rgba, [0.0, 1.0, 0.0, 0.4], atol=1e-7)
        assert int(arrow.type) == int(self.mujoco.mjtGeom.mjGEOM_ARROW)
        # mjv_connector: pos = start, z axis = direction, size[2] = length
        np.testing.assert_allclose(arrow.pos, [0, 0, 0.5], atol=1e-6)
        np.testing.assert_allclose(arrow.mat.reshape(3, 3)[:, 2], [0, 0, 1], atol=1e-6)
        assert arrow.size[2] == pytest.approx(0.2, abs=1e-6)

    def test_frame_draws_rgb_triad(self) -> None:
        scene = self._scene()
        overlays = [[DebugPrimitive(kind="frame", pos=(0, 0, 0), size=(0.1,))]]
        self.render_many.append_debug_primitives(
            scene, overlays, offsets=None, mesh_ids={}
        )
        assert scene.ngeom == 3
        for geom, expected_rgb in zip(
            scene.geoms[:3], ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
        ):
            assert int(geom.type) == int(self.mujoco.mjtGeom.mjGEOM_ARROW)
            np.testing.assert_allclose(geom.rgba[:3], expected_rgb, atol=1e-7)

    def test_grid_offset_applied(self) -> None:
        scene = self._scene()
        overlays = [
            None,
            [DebugPrimitive(kind="sphere", pos=(0, 0, 0.5), size=(0.05,))],
        ]
        offsets = np.array([[0.0, 0.0], [1.0, 2.0]])
        self.render_many.append_debug_primitives(
            scene, overlays, offsets=offsets, mesh_ids={}
        )
        assert scene.ngeom == 1
        np.testing.assert_allclose(scene.geoms[0].pos, [1.0, 2.0, 0.5], atol=1e-7)

    def test_text_is_documented_noop(self) -> None:
        scene = self._scene()
        overlays = [[DebugPrimitive(kind="text", pos=(0, 0, 0), text="label")]]
        self.render_many.append_debug_primitives(
            scene, overlays, offsets=None, mesh_ids={}
        )
        assert scene.ngeom == 0

    def test_ghost_geom_uses_mesh_dataid(self, tmp_path: Path) -> None:
        scene = self._scene()
        overlays = [
            [
                DebugPrimitive(
                    kind="ghost_geom", pos=(0, 0, 0.3), size=(2.0,), mesh_asset="goal.stl",
                    rgba=(0.2, 0.6, 1.0, 0.3),
                )
            ]
        ]
        self.render_many.append_debug_primitives(
            scene, overlays, offsets=None, mesh_ids={"goal.stl": 5}
        )
        assert scene.ngeom == 1
        geom = scene.geoms[0]
        assert int(geom.type) == int(self.mujoco.mjtGeom.mjGEOM_MESH)
        assert geom.dataid == 5
        np.testing.assert_allclose(geom.size, [2.0, 2.0, 2.0], atol=1e-7)

    def test_ghost_geom_unresolved_asset_fails(self) -> None:
        scene = self._scene()
        overlays = [[DebugPrimitive(kind="ghost_geom", pos=(0, 0, 0), mesh_asset="missing.stl")]]
        with pytest.raises(ValueError, match="missing.stl"):
            self.render_many.append_debug_primitives(
                scene, overlays, offsets=None, mesh_ids={}
            )

    def test_ghost_mesh_injection_into_xml_model(self, tmp_path: Path) -> None:
        obj_path = tmp_path / "goal.obj"
        obj_path.write_text(
            "v 0 0 0\nv 0.1 0 0\nv 0 0.1 0\nv 0 0 0.1\n"
            "f 1 3 2\nf 1 2 4\nf 2 3 4\nf 3 1 4\n"
        )
        model_path = tmp_path / "scene.xml"
        model_path.write_text(
            "<mujoco><worldbody><geom type='box' size='0.05 0.05 0.05'/></worldbody></mujoco>"
        )
        new_path, name_map = self.render_many._inject_ghost_mesh_assets(
            str(model_path), [str(obj_path)], tmp_path
        )
        assert new_path != str(model_path)
        model = self.mujoco.MjModel.from_binary_path(new_path)
        mesh_name = name_map[str(obj_path)]
        assert (
            self.mujoco.mj_name2id(model, self.mujoco.mjtObj.mjOBJ_MESH, mesh_name) >= 0
        )

    def test_ghost_mesh_registered_name_passes_through(self, tmp_path: Path) -> None:
        obj_path = tmp_path / "goal.obj"
        obj_path.write_text(
            "v 0 0 0\nv 0.1 0 0\nv 0 0.1 0\nv 0 0 0.1\n"
            "f 1 3 2\nf 1 2 4\nf 2 3 4\nf 3 1 4\n"
        )
        model_path = tmp_path / "scene.xml"
        model_path.write_text(
            f"<mujoco><asset><mesh name='goal' file='{obj_path}'/></asset>"
            "<worldbody><geom type='mesh' mesh='goal'/></worldbody></mujoco>"
        )
        new_path, name_map = self.render_many._inject_ghost_mesh_assets(
            str(model_path), ["goal"], tmp_path
        )
        assert new_path == str(model_path)
        assert name_map == {"goal": "goal"}

    def test_ghost_mesh_missing_file_fails(self, tmp_path: Path) -> None:
        model_path = tmp_path / "scene.xml"
        model_path.write_text(
            "<mujoco><worldbody><geom type='box' size='0.05 0.05 0.05'/></worldbody></mujoco>"
        )
        with pytest.raises(ValueError, match="neither a mesh name"):
            self.render_many._inject_ghost_mesh_assets(
                str(model_path), [str(tmp_path / "absent.obj")], tmp_path
            )

    def test_ghost_mesh_mjb_model_fails_closed(self, tmp_path: Path) -> None:
        mjb_path = tmp_path / "scene.mjb"
        self.mujoco.mj_saveModel(self.model, str(mjb_path))
        with pytest.raises(ValueError, match="compiled .mjb"):
            self.render_many._inject_ghost_mesh_assets(
                str(mjb_path), ["not_registered"], tmp_path
            )


class TestHeadlessPrimitiveRendering:
    """Offline snapshot rendering with overlays (skipped without a GL backend).

    Rendering runs in a clean subprocess (like the module's GL probe) so a
    driver-level EGL crash cannot take down the pytest process.
    """

    MODEL = """<mujoco>
      <worldbody>
        <light pos='0 0 3' dir='0 0 -1'/>
        <geom type='plane' size='5 5 0.1'/>
        <body name='base' pos='0 0 0.5'>
          <joint name='slide' type='slide' axis='1 0 0'/>
          <geom type='box' size='0.05 0.05 0.05'/>
        </body>
      </worldbody>
    </mujoco>"""

    OBJ = (
        "v 0 0 0\nv 0.1 0 0\nv 0 0.1 0\nv 0 0 0.1\n"
        "f 1 3 2\nf 1 2 4\nf 2 3 4\nf 3 1 4\n"
    )

    _SCRIPT = textwrap.dedent(
        """
        import sys
        import numpy as np
        from unisim.backend.base import DebugPrimitive
        from unisim.visualization import render_many

        model_path, obj_path, mode = sys.argv[1:4]
        states = [
            np.array([[0.0, 0.0, 0.0], [0.0, 0.1, 0.0]], dtype=np.float32)
            for _ in range(2)
        ]

        if mode == "overlay_diff":
            overlays = [
                [DebugPrimitive(kind="sphere", pos=(0.2, 0.0, 0.6), size=(0.05,))],
                [DebugPrimitive(kind="frame", pos=(0.0, 0.2, 0.5), size=(0.15,))],
            ]
            baseline = render_many.render_states_get_frames(
                states, model_path, width=160, height=120, num_processes=1
            )
            with_overlay = render_many.render_states_get_frames(
                states, model_path, width=160, height=120, num_processes=1,
                debug_overlays_list=[overlays] * len(states),
            )
            assert len(with_overlay) == len(states) == len(baseline)
            assert with_overlay[0].shape == (120, 160, 3)
            diff = np.abs(with_overlay[0].astype(int) - baseline[0].astype(int)).sum()
            assert diff > 0
        elif mode == "ghost":
            overlays = [
                [DebugPrimitive(
                    kind="ghost_geom", pos=(0.3, 0.0, 0.6), size=(1.0,),
                    mesh_asset=obj_path, rgba=(0.2, 0.6, 1.0, 0.5),
                )],
                None,
            ]
            frames = render_many.render_states_get_frames(
                states[:1], model_path, width=160, height=120, num_processes=1,
                cam_fov=45.0, debug_overlays_list=[overlays],
            )
            assert len(frames) == 1 and frames[0].shape == (120, 160, 3)
        elif mode == "tracking":
            overlays = [[DebugPrimitive(kind="arrow", pos=(0, 0, 0.8), size=(0.2,))], None]
            frames = render_many.render_states_get_frames_tracking(
                states[:1], model_path, width=160, height=120,
                tracking_env_idx=0, max_extra_envs=1, debug_overlays_list=[overlays],
            )
            assert len(frames) == 1 and frames[0].shape == (120, 160, 3)
        else:
            raise AssertionError(f"unknown mode {mode!r}")
        print("RENDER-OK")
        """
    )

    @pytest.fixture(autouse=True)
    def _render(self, tmp_path: Path):
        pytest.importorskip("mujoco")
        from unisim.visualization import render_many

        if not render_many.render_backend_usable():
            pytest.skip("no usable MuJoCo off-screen GL backend on this host")
        self._model_path = tmp_path / "scene.xml"
        self._model_path.write_text(self.MODEL)
        self._obj_path = tmp_path / "goal.obj"
        self._obj_path.write_text(self.OBJ)

    def _run_render_mode(self, mode: str) -> None:
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                self._SCRIPT,
                str(self._model_path),
                str(self._obj_path),
                mode,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0 and "RENDER-OK" in result.stdout, (
            f"render subprocess failed:\n{result.stdout}\n{result.stderr}"
        )

    def test_primitives_change_rendered_frames(self) -> None:
        self._run_render_mode("overlay_diff")

    def test_ghost_geom_renders_headless(self) -> None:
        self._run_render_mode("ghost")

    def test_tracking_render_with_overlay(self) -> None:
        self._run_render_mode("tracking")


class TestAppendDebugPrimitivesPublic:
    """Public single-callable entry shared by workers and interactive viewers."""

    @pytest.fixture(autouse=True)
    def _mujoco(self):
        self.mujoco = pytest.importorskip("mujoco")
        from unisim.visualization import render_many

        self.render_many = render_many
        self.model = self.mujoco.MjModel.from_xml_string(
            "<mujoco><worldbody><geom type='box' size='0.05 0.05 0.05'/></worldbody></mujoco>"
        )

    def test_single_env_no_offsets_returns_count(self) -> None:
        scene = self.mujoco.MjvScene(self.model, maxgeom=16)
        overlays = [
            [
                DebugPrimitive(kind="sphere", pos=(0, 0, 0.5), size=(0.05,)),
                DebugPrimitive(kind="frame", pos=(0, 0, 0), size=(0.1,)),
                DebugPrimitive(kind="text", pos=(0, 0, 0), text="skip-me"),
            ]
        ]
        added = self.render_many.append_debug_primitives(scene, overlays)
        assert added == 4  # sphere + 3 frame axes; text is a documented no-op
        assert scene.ngeom == 4

    def test_none_overlays_injects_nothing(self) -> None:
        scene = self.mujoco.MjvScene(self.model, maxgeom=16)
        assert self.render_many.append_debug_primitives(scene, None) == 0
        assert scene.ngeom == 0

    def test_maxgeom_caps_injection(self) -> None:
        scene = self.mujoco.MjvScene(self.model, maxgeom=2)
        overlays = [
            [
                DebugPrimitive(kind="sphere", pos=(0, 0, 0), size=(0.05,)),
                DebugPrimitive(kind="sphere", pos=(0.1, 0, 0), size=(0.05,)),
                DebugPrimitive(kind="sphere", pos=(0.2, 0, 0), size=(0.05,)),
            ]
        ]
        assert self.render_many.append_debug_primitives(scene, overlays) == 2
        assert scene.ngeom == 2

    def test_ghost_geom_unregistered_mesh_fails_closed(self) -> None:
        scene = self.mujoco.MjvScene(self.model, maxgeom=16)
        overlays = [[DebugPrimitive(kind="ghost_geom", pos=(0, 0, 0), mesh_asset="goal.stl")]]
        with pytest.raises(ValueError, match="registered in the model"):
            self.render_many.append_debug_primitives(scene, overlays, mesh_ids={})


class TestInteractiveOverlayInjection:
    """mjwarp passive-viewer scene injection (no GL context required)."""

    OBJ = (
        "v 0 0 0\nv 0.1 0 0\nv 0 0.1 0\nv 0 0 0.1\n"
        "f 1 3 2\nf 1 2 4\nf 2 3 4\nf 3 1 4\n"
    )

    @pytest.fixture(autouse=True)
    def _mujoco(self, tmp_path: Path):
        self.mujoco = pytest.importorskip("mujoco")
        from unisim.backend.mjwarp.playback import _inject_interactive_debug_overlays

        self.inject = _inject_interactive_debug_overlays
        obj_path = tmp_path / "goal.obj"
        obj_path.write_text(self.OBJ)
        self.model = self.mujoco.MjModel.from_xml_string(
            f"<mujoco><asset><mesh name='goal' file='{obj_path}'/></asset>"
            "<worldbody><geom type='mesh' mesh='goal'/></worldbody></mujoco>"
        )

    def _scene(self, maxgeom: int = 16):
        return self.mujoco.MjvScene(self.model, maxgeom=maxgeom)

    def test_none_overlays_clears_scene(self) -> None:
        scene = self._scene()
        overlays = [[DebugPrimitive(kind="sphere", pos=(0, 0, 0.5), size=(0.05,))]]
        added = self.inject(
            user_scn=scene, overlays=overlays, world=0, num_envs=1,
            model=self.model, mesh_id_cache={},
        )
        assert added == 1 and scene.ngeom == 1
        # A frame without overlays resets ngeom (sync() does not clear user geoms).
        assert self.inject(
            user_scn=scene, overlays=None, world=0, num_envs=1,
            model=self.model, mesh_id_cache={},
        ) == 0
        assert scene.ngeom == 0

    def test_only_tracked_world_is_injected(self) -> None:
        scene = self._scene()
        overlays = [
            [DebugPrimitive(kind="sphere", pos=(0, 0, 0.5), size=(0.05,))],
            [DebugPrimitive(kind="frame", pos=(0, 0, 0.2), size=(0.1,))],
        ]
        added = self.inject(
            user_scn=scene, overlays=overlays, world=1, num_envs=2,
            model=self.model, mesh_id_cache={},
        )
        assert added == 3  # the frame triad of env 1 only
        assert scene.ngeom == 3
        np.testing.assert_allclose(scene.geoms[0].pos, [0, 0, 0.2], atol=1e-7)

    def test_empty_tracked_world_injects_nothing(self) -> None:
        scene = self._scene()
        overlays = [[DebugPrimitive(kind="sphere", pos=(0, 0, 0.5), size=(0.05,))], None]
        assert self.inject(
            user_scn=scene, overlays=overlays, world=1, num_envs=2,
            model=self.model, mesh_id_cache={},
        ) == 0
        assert scene.ngeom == 0

    def test_ghost_geom_resolves_registered_mesh_and_caches_id(self) -> None:
        scene = self._scene()
        cache: dict[str, int] = {}
        overlays = [[DebugPrimitive(kind="ghost_geom", pos=(0, 0, 0.3), mesh_asset="goal")]]
        added = self.inject(
            user_scn=scene, overlays=overlays, world=0, num_envs=1,
            model=self.model, mesh_id_cache=cache,
        )
        mesh_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_MESH, "goal")
        assert added == 1
        assert cache == {"goal": mesh_id}
        assert scene.geoms[0].dataid == mesh_id

    def test_ghost_geom_unregistered_mesh_fails_closed(self) -> None:
        scene = self._scene()
        overlays = [[DebugPrimitive(kind="ghost_geom", pos=(0, 0, 0), mesh_asset="absent")]]
        with pytest.raises(ValueError, match="not registered in the playback model"):
            self.inject(
                user_scn=scene, overlays=overlays, world=0, num_envs=1,
                model=self.model, mesh_id_cache={},
            )

    def test_wrong_outer_length_fails(self) -> None:
        scene = self._scene()
        with pytest.raises(ValueError, match=r"len == 2"):
            self.inject(
                user_scn=scene, overlays=[[]], world=0, num_envs=2,
                model=self.model, mesh_id_cache={},
            )


class TestOnFrameCallback:
    def test_none_passthrough(self) -> None:
        from unisim.backend.playback_common import apply_on_frame_callback

        frames = [np.zeros((4, 4, 3), dtype=np.uint8)]
        assert apply_on_frame_callback(frames, None, backend_label="test") is frames

    def test_replacement_and_keep(self) -> None:
        from unisim.backend.playback_common import apply_on_frame_callback

        frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(3)]
        calls: list[int] = []

        def on_frame(index: int, frame: np.ndarray) -> np.ndarray | None:
            calls.append(index)
            if index == 1:
                return np.full_like(frame, 255)
            return None

        out = apply_on_frame_callback(frames, on_frame, backend_label="test")
        assert calls == [0, 1, 2]
        assert out[0] is frames[0] and out[2] is frames[2]
        assert out[1].min() == 255

    def test_bad_replacement_fails_closed(self) -> None:
        from unisim.backend.playback_common import apply_on_frame_callback

        frames = [np.zeros((4, 4, 3), dtype=np.uint8)]
        with pytest.raises(ValueError, match="on_frame must return None"):
            apply_on_frame_callback(
                frames, lambda i, f: np.zeros((2, 2, 3), dtype=np.uint8), backend_label="test"
            )
        with pytest.raises(ValueError, match="dtype"):
            apply_on_frame_callback(
                frames, lambda i, f: f.astype(np.float32), backend_label="test"
            )

    @pytest.mark.parametrize(
        ("module_path", "class_name"),
        [
            ("unisim.backend.motrix.backend", "MotrixBackend"),
            ("unisim.backend.genesis.backend", "GenesisBackend"),
            ("unisim.backend.subprocess_ipc.backend", "MjcfSubprocessBackend"),
        ],
    )
    def test_native_renderer_backends_fail_closed(
        self, module_path: str, class_name: str
    ) -> None:
        import importlib

        backend_cls = getattr(importlib.import_module(module_path), class_name)
        backend = backend_cls.__new__(backend_cls)
        with pytest.raises(NotImplementedError, match="on_frame"):
            backend.run_playback(
                env=None,
                initialize=lambda: None,
                step=lambda o: o,
                num_steps=1,
                on_frame=lambda i, frame: None,
            )

    def test_mjwarp_interactive_fails_closed_on_on_frame(self) -> None:
        from unisim.backend.mjwarp.playback import run_mjwarp_playback

        with pytest.raises(NotImplementedError, match="on_frame"):
            run_mjwarp_playback(
                backend=None,
                env=None,
                initialize=lambda: None,
                step=lambda o: o,
                num_steps=1,
                output_video=None,
                render_spacing=None,
                headless=False,
                record_video=False,
                snapshot_shape=(1, 2),
                frame_state_getter=None,
                camera_kwargs=None,
                on_frame=lambda i, frame: None,
            )

    def test_newton_interactive_fails_closed_on_on_frame(self) -> None:
        from unisim.backend.newton.backend import NewtonBackend

        backend = NewtonBackend.__new__(NewtonBackend)
        with pytest.raises(NotImplementedError, match="on_frame"):
            backend.run_playback(
                env=None,
                initialize=lambda: None,
                step=lambda o: o,
                num_steps=1,
                headless=False,
                record_video=False,
                on_frame=lambda i, frame: None,
            )


class TestOnFrameEndToEnd:
    """Offline MuJoCo playback applies on_frame before video encoding."""

    _SCRIPT = textwrap.dedent(
        """
        import sys
        from types import SimpleNamespace
        import imageio.v2 as imageio
        import numpy as np
        from unisim.backend.mujoco.playback import run_mujoco_playback
        from unisim.scene import SceneCfg

        model_path, out_path = sys.argv[1:3]
        env = SimpleNamespace(
            cfg=SimpleNamespace(scene=SceneCfg(model_file=model_path), ctrl_dt=0.05)
        )
        state = np.array([[0.0, 0.0, 0.0], [0.0, 0.1, 0.0]], dtype=np.float32)
        calls = []

        def on_frame(index, frame):
            calls.append(index)
            assert frame.dtype == np.uint8 and frame.ndim == 3
            painted = frame.copy()
            painted[..., 0] = 255
            return painted

        result = run_mujoco_playback(
            env=env,
            initialize=lambda: None,
            step=lambda obs: obs,
            num_steps=2,
            output_video=out_path,
            render_spacing=1.0,
            headless=True,
            record_video=True,
            frame_state_getter=lambda: state,
            camera_kwargs=None,
            on_frame=on_frame,
        )
        assert result == out_path
        assert calls == [0, 1]
        frames = list(imageio.mimread(out_path))
        assert len(frames) == 2
        assert frames[0][..., 0].min() == 255
        print("ONFRAME-OK")
        """
    )

    def test_on_frame_paints_video_frames(self, tmp_path: Path) -> None:
        pytest.importorskip("mujoco")
        pytest.importorskip("imageio")
        from unisim.visualization import render_many

        if not render_many.render_backend_usable():
            pytest.skip("no usable MuJoCo off-screen GL backend on this host")
        model_path = tmp_path / "scene.xml"
        model_path.write_text(TestHeadlessPrimitiveRendering.MODEL)
        out_path = tmp_path / "play.gif"
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-c", self._SCRIPT, str(model_path), str(out_path)],
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert result.returncode == 0 and "ONFRAME-OK" in result.stdout, (
            f"on_frame subprocess failed:\n{result.stdout}\n{result.stderr}"
        )
