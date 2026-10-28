# ADR：packed reset 生命周期事务

[English](../en/adr-packed-reset-lifecycle.md) | [中文](adr-packed-reset-lifecycle.md)

## 状态

已在 1.x 契约线接受（首个切片，#358）。

## 背景

UniLab 完成 Torch-only Manager 收尾后，`ResetStateTransaction` 是最后一个结构性 NumPy reset 组合器。它维护 host mask/buffer、合并选中行状态与不相交列的模型字段写入、记录已提交 baseline，最后调用底层 `SimBackend.set_state_tensor()` 边界。该组合职责属于 UniSim，而不属于 task 或 Manager 代码。

现有 tensor 契约已提供正确的提交原语：选中行 qpos/qvel、device-resident 最终值模型字段 scatter、持久通道清理、time reset 以及 `AUTHORITATIVE_VIEWS` 发布。缺失的是在不引入 host mirror 的情况下组合最终值并持有已提交 baseline 的事务层。

## 决策

UniSim 拥有 backend-neutral 的 staged packed reset 事务：

- `SimBackend.bind_packed_reset(device=None)` 返回 backend-owned `PackedResetTransaction`，否则快速失败。
- `TensorLifecycleCapabilities` 新增 `packed_reset`、`packed_reset_fields` 与 `packed_reset_mocap`。不支持的生命周期不得声明它们。
- 事务持有全宽、device-resident 的已提交 baseline，并由 backend 的权威（fixed-variant aware）`get_reset_term_default()` 初始化。
- 固定宽度路径保持同样的全宽 staging：`begin_state_mask(mask)`、`write_state_mask(mask, ...)` 与 `commit_state_mask(mask, ...)` 消费全宽 bool mask，保留 false rows，并把 selected-row 语义委托给公开 backend mask 边界。这样避免 owner 侧 `nonzero()` 压缩，同时不改变上述 staged-row 契约。
- `write_model_field(field, rows, values, columns=None)` 合并最终绝对值的选中行写入，并支持 PD `kp`/`kd` 等不相交选中列合并。
- `commit_state(rows, qpos, qvel)` 通过携带 staged 选中行表调用一次 `set_state_tensor()`，形成唯一公开 reset 边界；仅对本次提交行、且仅当该边界成功返回后才提升 baseline；其他行的 staged 写入保留到 `discard()`。
- `discard(field=None)` 在提交前显式丢弃 staged 写入。
- 事务同时拥有全宽 device 状态 staging：`set_state_default()` 安装权威单行默认值，`begin_state(rows)` 为选中行恢复默认，`write_state(...)` 合并选中列，`commit(rows)` 只提交 staged 行。由此保持 selected-reset 精确性（`reset rows == written rows`），且不需要 host mask。
- 持久通道（`ctrl`、`act`、`qfrc_applied`、`xfrc_applied`、`qacc_warmstart`）与 `time` 沿用底层 tensor 边界的既有 reset 语义。
- `write_mocap_pose(body_name, rows, poses)` stage 选中行 xyz/单位 wxyz pose。组合提交先应用 generalized state，再应用 mocap pose；`commit_mocap_poses()` 提供既有 mocap-only 边界，不重置 generalized state。

### 适配器行为

- MuJoCo（`HOST_BRIDGE`）在同一显式 tensor-to-host reset 边界内接受 selected-row 与 full-width mask 形态的 `TensorResetRandomizationPayload`，先同步一次再转换为 adapter 私有的 host 数组。持久的 `HostBridgeTransferPlan.apply_reset()` 仍是 selected-row ID 的单包快速路径。
- MotrixSim（`HOST_BRIDGE`）在其运行时协商的字段子集内接受同样的 selected-row 与 full-width mask 形态，在 host reset 边界内转换 payload，并权威发布选中行且没有单独 full-refresh 阶段。
- MJWarp（`DEVICE_RESIDENT`）在 device 上组合，并通过既有 device-side scatter 提交，发布语义为 `AUTHORITATIVE_VIEWS`。`geom_size` 因派生 bounds 目前在 host 计算而在 device 平面继续排除。
- Genesis 与 Newton（`DEVICE_RESIDENT`）用窄的最终值字段子集提交 selected-row 状态：Genesis 在[统一 Genesis DR 生命周期](adr-genesis-dr-lifecycle.md)下于 whole-MJCF 与 portable 场景暴露 `body_mass`、`kp`、`kd`；Newton 通过 MuJoCo-family actuator array 暴露 `kp`、`kd`。Newton 使用 SolverMuJoCo 显式的 control-source/index 映射，把公开的 motor、position、velocity 执行器行解析到原生行；映射缺失、存在歧义或不构成排列时快速失败。
- IsaacGym 与 IsaacSim（`EXTERNAL_WORKER` / `CUDA_IPC`）把 selected-row `kp`/`kd` stage 到 device arena。worker 拥有一个显式命名的 reset-only property bridge，用于 CPU-native engine setter；step/state/control/sensor 热路径绝不进入该 bridge。backend-owned mask boundary 可以在公开 `set_state_tensor_mask()` 契约之后自行压缩 true rows。
- MuJoCo 为 kinematic entity root 声明 packed reset mocap 支持；选中 Torch 行与 pose 只在提交时通过显式 host 边界。本阶段 MJWarp 的 mocap 写入保持快速失败。
- 不支持字段、fixed variants、model-file tensor lane 与其余适配器保持快速失败；不进行 backend-private probing。

### 显式排除

- owner-subset 动态行 API（例如 motion-wrap mutation）保持独立 owner API，不重载 reset 行集合。

### Delta 项

- 自 UniLab issue #2410 起，`base_mass_delta` 与 `base_com_offset` 是单一 torch payload 家族的一等字段；原先的 host-only 限制与 `PACKED_RESET_HOST_ONLY_FIELDS` 已移除。transaction stage 采样得到的 delta，并在 commit 时与权威 baseline（由冷 NumPy metadata leaf `get_reset_term_default()` 提供）在 torch 中合成为最终绝对的 `body_mass` / `body_ipos` 表：`final = baseline + delta`。adapter 内部的 delta merge 可作为 adapter 私有实现保留，但公开 payload 不再限制 delta 字段。

## 影响

- 下游 Manager 可以删除模型字段 reset 组合中的 NumPy mask/buffer 与已提交 baseline 数组。
- 提交失败不会污染后续 baseline，因为提升只发生在成功之后。
- 模型字段变更沿用既有 fixed-graph 失效语义；公开提交返回后，状态与 sensor view 立即权威。
- 整个事务在 adapter reset timing 之外额外报告 `packed_reset_compose_ms`、`packed_reset_commit_ms` 与 `packed_reset_total_ms`。

## 后续工作

- 与 multi-step/CUDA-graph 执行组合，同时保证 graph-unsafe 字段变更在 graph 失效下快速失败。
