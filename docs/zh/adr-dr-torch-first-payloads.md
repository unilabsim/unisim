# ADR：torch-first 域随机化 Payload

[English](../en/adr-dr-torch-first-payloads.md) | [中文](adr-dr-torch-first-payloads.md)

## 状态

已接受（2026-10-19，UniLab issue #2410）。

## 背景

全栈 DR 审计（UniLab `DR_AUDIT.md`，2026-10-19）发现公开域随机化契约携带两套并行的 payload 载体：NumPy `ResetRandomizationPayload` 服务 host 平面，`TensorResetRandomizationPayload` 服务 device-resident 适配器，`require_op_host_payload` 则给 interval op 提供了公开的 host payload 契约。同一个 `push` interval 项有两种 payload 语义——host 适配器上的 payload 是采样范围、由后端内部全局 RNG（`np.random`）采样，而 MJWarp device 路径上是生产方采样的最终力表——导致 Manager seed 无法约束 host push。`base_mass_delta` 与 `base_com_offset` 是 host-only 字段，被排除在 device payload 之外（`PACKED_RESET_HOST_ONLY_FIELDS`），迫使 owner 预先把 delta 合成为绝对值表。`DomainRandomizationCapabilities` 每个适配器只声明单一 term 集合，掩盖了 host 与 device 平面支持不同子集的事实（MJWarp `geom_size`、Isaac host-mapped 与 CUDA IPC 两条通道）。后端 conformance 从不断言声明的 reset term 真正改写了 model field。UniLab 一侧，CUDA 采样的 reset 值在提交前还要经过 CPU tensor 强制，付出一次 D2H→H2D 往返。

## 决策

公开 DR 契约改为 torch-first / torch-only：

- 从公开契约中移除 `ResetRandomizationPayload`（NumPy）与 `require_op_host_payload`。`TensorResetRandomizationPayload` 是唯一公开的 reset payload；类名保持不变，并新增 `base_mass_delta` / `base_com_offset` 字段，覆盖全部 16 个 reset term。`IntervalTermOp.payload` 与 `TensorIntervalPayload` 是 torch-first 的 interval 载体；`body_ids` 同样 torch-first。
- payload 字段保持 `Any` 类型注解与惰性 torch 校验（`validate_tensor_reset_payload`、`validate_interval_op`），使 `unisim-core` 维持仅依赖 NumPy 的打包契约：base 包绝不在模块顶层 import torch。
- payload tensor 的设备决定执行平面：`torch.cpu` 为 host 平面，CUDA 设备为 device 平面。不引入新的配置参数。
- `DomainRandomizationCapabilities` 按平面拆分：`supported_reset_terms` / `supported_interval_terms` 声明 host 平面，新增的 `supported_device_reset_terms` / `supported_device_interval_terms` 声明 device 平面。空集合 fail-closed；在未声明某 term 的平面上提交该 term 时点名 term 快速失败。
- delta 项不再仅限 host。packed reset transaction stage `base_mass_delta` / `base_com_offset`，并在 commit 时与权威 baseline（由冷 NumPy metadata leaf `get_reset_term_default()` 提供）在 torch 中合成为最终绝对的 `body_mass` / `body_ipos` 表（`final = baseline + delta`）。`PACKED_RESET_HOST_ONLY_FIELDS` 已删除。
- push 在所有后端统一为生产方采样，沿用 MJWarp device 模式。后端内部的全局 RNG push 采样（MuJoCo、Motrix）已删除；host 适配器缺少生产方采样的 push payload 时快速失败。
- NumPy 仅保留在显式边界之后：host 边界适配器（MuJoCo、Motrix、IsaacGym、IsaacSim、subprocess IPC）使用的 adapter 私有 `unisim.dr._host` 转换 helper、`get_reset_term_default()` 等冷元数据，以及 serialization/playback leaf。
- 后端 conformance 长牙：每个声明 reset randomization 的适配器必须对至少一个声明的 reset term 完成可验证的往返——写入非 baseline 值、commit、在新事务中读回 baseline 并断言 allclose。

## 影响

- 构造 NumPy reset 或 interval payload 的消费者必须改为构造 torch tensor；由 payload 设备而非配置开关选择 host 或 device 平面。CPU tensor payload 在 host 平面适配器上的行为不变。
- UniLab 一侧 CUDA 驻留的 reset DR 不再付出 D2H→H2D 往返；event term 直接提交 device tensor。
- 按平面拆分的能力声明使此前被掩盖的平面差异（例如 MJWarp `geom_size`）显式化，并逐项 fail-closed。
- adapter 私有的 NumPy 转换集中在 `unisim.dr._host` 并保持私有：不进入 `__all__`，也不构成 Manager 代码的公开 fallback。
- Genesis 不是 host 边界适配器，不 import `unisim.dr._host`；其统一的 runtime DR 生命周期在 backend 绑定的 CUDA device 平面上直接消费 Torch tensor。
- 声明但无效的 reset term 会被 conformance 捕获，而不是靠下游观察漂移才发现。

## 相关决策

- [tensor 生命周期 ADR](adr-tensor-lifecycle.md) 定义该 payload 家族接入的可选、快速失败 tensor 生命周期。
- [packed reset 生命周期 ADR](adr-packed-reset-lifecycle.md) 定义组合并提交 torch payload 的事务。
- [统一 Genesis DR 生命周期 ADR](adr-genesis-dr-lifecycle.md) 把 device 平面契约统一应用到 Genesis whole-MJCF 与 portable 场景。
- [legacy NumPy removal ADR](adr-legacy-numpy-removal.md) 定义其余公开 NumPy leaf。
- UniLab [ADR-0015：无 Numba runtime 与显式 NumPy leaf 边界](https://github.com/Motphys/UniLab/blob/main/docs/sphinx/source/adr/ADR-0015-no-numba-and-explicit-numpy-leaf-boundaries.md) 在消费侧约束 NumPy 只能位于显式 leaf owner。
