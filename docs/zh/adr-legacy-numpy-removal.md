# ADR：移除 legacy NumPy SimBackend 生命周期

[English](../en/adr-legacy-numpy-removal.md) | [中文](adr-legacy-numpy-removal.md)

## 状态

已在 `unisim-core` 2.0.0 接受（UniLab issue #2280）。

## 背景

1.x 的 `SimBackend` 接口来自 tensor 化之前的 UniLab 运行时，暴露 NumPy step、整批 reset、generalized-state 快照、base/DoF/body getter 与具名 sensor getter。UniLab Manager 运行时已不再消费该 surface；已接受的 tensor 生命周期提供公共 state、sensor、body、step、selected-reset、packed host-bridge 与 packed reset 边界。

同时保留两套 surface 会让未支持适配器看起来拥有 host fallback，并迫使 task 代码维护两个权威 carrier。这也与 UniLab ADR-0015 冲突：NumPy 只允许位于显式 adapter、冷 metadata、serialization 或 playback leaf 内。

## 决策

2.0.0 直接移除 legacy public training lifecycle，不提供 alias。1.7.15 是最后一个包含该 surface 的版本。Concrete CPU adapter 可以保留 `_legacy_*` 方法作为私有 host-format 实现细节，但它们不属于 `SimBackend` 类型契约或适配器支持声明。

| 1.x 移除面 | 2.0.0 替代 |
| --- | --- |
| `step(np.ndarray)` | `step_tensor(torch.Tensor, nsteps=...)` |
| `set_state(...)` 与 NumPy reset 组合 | `set_state_tensor(...)`；需要 staging、选中列合并、baseline、模型字段或 mocap 时使用 `begin_packed_reset(...)` |
| `get_state()`、`get_base_*`、`get_dof_pos/vel` | `get_state_views(...)` |
| `get_sensor_data*` | `get_sensor_view(...)`、`bind_sensor_data(...)` 或已协商 sensor packet |
| Body getter 与 body-state copy helper | `get_tracked_body_views(...)` 或公共 tensor sensor projection |
| 零散 CPU host exchange | `compile_host_bridge_io(...)` 及其 plan boundary |

NumPy 仍保留在显式 scoped public leaf 中：冷 model/default/layout metadata 与索引表、声明式 interval-DR 输入和 pre-step callback 格式、mapped entity transaction、playback/render API、physics-state snapshot 与 serialization。这些 leaf 不恢复 NumPy 训练生命周期，也不得成为 tensor-capable Manager 代码的隐藏 fallback。

仓库同时 gate base 契约与 concrete backend 类，防止被移除的方法名重新出现。

## 影响

- 链接到 1.x 方法的消费者会在 attribute lookup 失败，而不会静默转换 carrier。
- Tensor 支持必须通过 `TensorLifecycleCapabilities` 协商；未支持生命周期快速失败。
- CPU-authoritative adapter 只在显式、可度量的 host-bridge 或 reset 边界转换。
- 即使训练生命周期 tensor-only，playback 与 rendering 仍可独立声明支持。

## 相关文档

- [Tensor 生命周期](adr-tensor-lifecycle.md)
- [Packed reset 生命周期](adr-packed-reset-lifecycle.md)
- UniLab [ADR-0015：无 Numba runtime 与显式 NumPy leaf 边界](https://github.com/Motphys/UniLab/blob/main/docs/sphinx/source/adr/ADR-0015-no-numba-and-explicit-numpy-leaf-boundaries.md)
