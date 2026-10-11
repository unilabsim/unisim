# ADR：packed reset 生命周期事务

[English](../en/adr-packed-reset-lifecycle.md) | [中文](adr-packed-reset-lifecycle.md)

## 状态

已在 1.x 契约线接受（首个切片，#358）。

## 背景

UniLab 完成 Torch-only Manager 收尾后，`ResetStateTransaction` 是最后一个结构性 NumPy reset 组合器。它维护 host mask/buffer、合并选中行状态与不相交列的模型字段写入、记录已提交 baseline，最后调用底层 `SimBackend.set_state_tensor()` 边界。该组合职责属于 UniSim，而不属于 task 或 Manager 代码。

现有 tensor 契约已提供正确的提交原语：选中行 qpos/qvel、device-resident 最终值模型字段 scatter、持久通道清理、time reset 以及 `AUTHORITATIVE_VIEWS` 发布。缺失的是在不引入 host mirror 的情况下组合最终值并持有已提交 baseline 的事务层。

## 决策

UniSim 拥有 backend-neutral 的 staged packed reset 事务：

- `SimBackend.begin_packed_reset(device=None)` 返回 backend-owned `TorchPackedResetComposer`，否则快速失败。
- `TensorLifecycleCapabilities` 新增 `packed_reset`、`packed_reset_fields` 与 `packed_reset_mocap`。不支持的生命周期不得声明它们。
- 事务持有全宽、device-resident 的已提交 baseline，并由 backend 的权威（fixed-variant aware）`get_reset_term_default()` 初始化。
- `write_model_field(field, rows, values, columns=None)` 合并最终绝对值的选中行写入，并支持 PD `kp`/`kd` 等不相交选中列合并。
- `commit_state(rows, qpos, qvel)` 通过携带 staged 选中行表调用一次 `set_state_tensor()`，形成唯一公开 reset 边界；仅对本次提交行、且仅当该边界成功返回后才提升 baseline；其他行的 staged 写入保留到 `discard()`。
- `discard(field=None)` 在提交前显式丢弃 staged 写入。
- 事务同时拥有全宽 device 状态 staging：`set_state_default()` 安装权威单行默认值，`begin_state(rows)` 为选中行恢复默认，`write_state(...)` 合并选中列，`commit(rows)` 只提交 staged 行。由此保持 selected-reset 精确性（`reset rows == written rows`），且不需要 host mask。
- 持久通道（`ctrl`、`act`、`qfrc_applied`、`xfrc_applied`、`qacc_warmstart`）与 `time` 沿用底层 tensor 边界的既有 reset 语义。
- `write_mocap_pose(body_name, rows, poses)` stage 选中行 xyz/单位 wxyz pose。组合提交先应用 generalized state，再应用 mocap pose；`commit_mocap_poses()` 提供既有 mocap-only 边界，不重置 generalized state。

### 适配器行为

- MuJoCo（`HOST_BRIDGE`）在同一显式 tensor-to-host reset 边界内接受 `TensorResetRandomizationPayload`，先同步一次再物化 host payload。持久的 `HostBridgeTransferPlan.apply_reset()` 仍是单包快速路径。
- MJWarp（`DEVICE_RESIDENT`）在 device 上组合，并通过既有 device-side scatter 提交，发布语义为 `AUTHORITATIVE_VIEWS`。`geom_size` 因派生 bounds 目前在 host 计算而继续排除。
- MuJoCo 为 kinematic entity root 声明 packed reset mocap 支持；选中 Torch 行与 pose 只在提交时通过显式 host 边界。本阶段 MJWarp 的 mocap 写入保持快速失败。
- 其余适配器保持快速失败；不进行 backend-private probing。

### 显式排除

- `base_mass_delta` 与 `base_com_offset` 仍是 host-only payload 字段，但其 reset 语义不需要 device delta 表示：owner 先把权威 baseline 与采样 delta 合成为最终绝对的 `body_mass` / `body_ipos` device 表，再 stage 进事务。把 delta 字段本身传入 device payload 会快速失败；不引入隐藏 host fallback 或 adapter 私有派生。
- owner-subset 动态行 API（例如 motion-wrap mutation）保持独立 owner API，不重载 reset 行集合。

## 影响

- 下游 Manager 可以删除模型字段 reset 组合中的 NumPy mask/buffer 与已提交 baseline 数组。
- 提交失败不会污染后续 baseline，因为提升只发生在成功之后。
- 模型字段变更沿用既有 fixed-graph 失效语义；公开提交返回后，状态与 sensor view 立即权威。
- 整个事务在 adapter reset timing 之外额外报告 `packed_reset_compose_ms`、`packed_reset_commit_ms` 与 `packed_reset_total_ms`。

## 后续工作

- 扩展 base mass/COM 的最终值 device 表示。
- 与 multi-step/CUDA-graph 执行组合，同时保证 graph-unsafe 字段变更在 graph 失效下快速失败。
