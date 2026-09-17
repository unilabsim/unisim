"""Test-only framed IPC client that can exercise the actual isolated worker."""

from __future__ import annotations

import select
import subprocess
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np

from unisim.backend.isaacgym import worker
from unisim.backend.isaacgym.dependencies import build_worker_env, resolve_isaacgym_runtime
from unisim.backend.subprocess_ipc import protocol
from unisim.entity_state import prepare_scene_reset


class SceneClient:
    def __init__(self, payload, log_path: Path):
        runtime = resolve_isaacgym_runtime()
        self.layout = protocol.load_scene_layout(payload["scene_layout"])
        self.handles = {}
        self.slots = {}
        self.log = log_path.open("w")
        self.proc = subprocess.Popen(
            [str(runtime.python), worker.__file__, "--protocol", protocol.__file__],
            env=build_worker_env(runtime),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
        )
        try:
            self.meta = self.request(
                protocol.CMD_INIT, {**payload, "isaacgym_python": str(runtime.isaacgym_python)}
            )
            specs = {}
            for name, shape in protocol.scene_slot_shapes(payload["num_envs"], self.layout).items():
                handle = shared_memory.SharedMemory(
                    create=True, size=protocol.slot_allocation_nbytes(name, shape)
                )
                self.handles[name] = handle
                array = np.ndarray(shape, dtype=protocol.slot_dtype(name), buffer=handle.buf)
                array.fill(0)
                self.slots[name] = array
                specs[name] = {"shm": handle.name, "shape": list(shape), "dtype": str(array.dtype)}
            self.request(protocol.CMD_ATTACH, {"slots": specs})
        except BaseException:
            self.close()
            raise

    def request(self, command, payload=None):
        protocol.send_message(self.proc.stdin, command, payload)
        ready, _, _ = select.select([self.proc.stdout], [], [], 30)
        if not ready:
            self.proc.kill()
            self.proc.wait(timeout=10)
            raise TimeoutError("native scene worker did not answer " + command)
        reply = protocol.recv_message(self.proc.stdout)
        if reply["cmd"] == protocol.CMD_ERROR:
            raise RuntimeError(str(reply["payload"]))
        return reply.get("payload")

    def reset(self, request):
        prepared = prepare_scene_reset(
            self.layout,
            request,
            self.slots["qpos"],
            self.slots["qvel"],
            self.slots["entity_root_state"],
        )
        count = len(prepared.env_ids)
        for field, values in (
            ("reset_env_ids", prepared.env_ids),
            ("reset_qpos", prepared.qpos),
            ("reset_qvel", prepared.qvel),
            ("reset_entity_root_state", prepared.roots),
        ):
            self.slots[field][:count] = values
        for field, values in (
            ("reset_qpos_mask", prepared.qpos_mask),
            ("reset_qvel_mask", prepared.qvel_mask),
            ("reset_root_mask", prepared.root_mask),
        ):
            self.slots[field][:] = values
        return self.request(
            protocol.CMD_RESET_ENTITIES,
            {"count": count, "entity_names": list(prepared.entity_names)},
        )

    def close(self):
        if self.proc.poll() is None:
            try:
                self.request(protocol.CMD_SHUTDOWN)
                self.proc.wait(timeout=10)
            except Exception:
                self.proc.kill()
                self.proc.wait(timeout=10)
        for stream in (self.proc.stdin, self.proc.stdout):
            if stream is not None:
                try:
                    stream.close()
                except BrokenPipeError:
                    pass
        for handle in self.handles.values():
            handle.close()
            handle.unlink()
        self.handles.clear()
        self.log.close()
