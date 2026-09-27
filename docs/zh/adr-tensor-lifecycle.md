# ADR：可选后端 Tensor 生命周期

[English](../en/adr-tensor-lifecycle.md) | [中文](adr-tensor-lifecycle.md)

## 状态

已接受（2026-09-27）

## 背景

UniLab 的 Manager-Based 运行时与 collector 目前以 NumPy 为边界，而 GPU 物理引擎已经暴露稳定的批量设备数组。若用各引擎私有 tensor API 直接全局替换 NumPy，会破坏既有 manager、replay 与 playback 消费者。CPU 物理引擎在 manager 计算加速后也仍需可用。

## 决策

`SimBackend` 暴露可选且默认快速失败的 tensor 生命周期，而不是全局替换环境契约。`TensorExecution` 区分 `DEVICE_RESIDENT`、`HOST_BRIDGE` 与默认 `UNSUPPORTED`；`get_tensor_capabilities()` 暴露各方法和 reset 特性的部分支持边界。声明支持的适配器可实现 `get_state_views()`、`get_sensor_view()`、`step_tensor()` 与 `set_state_tensor()`。

基础包仍不依赖 Torch；适配器按需 lazy-import tensor 运行时，并拥有 stream、布局与传输语义。

Tensor 输入必须是连续 Torch tensor。control 是 float32 `(num_envs, num_actuators)`；reset row ID 是 `[0, num_envs)` 内唯一的 int64，qpos/qvel 是同一 accepted device 上的 float32 `(rows, nq/nv)`。生产方负责有限性检查，避免适配器在热路径强制标量同步；row 校验允许一次有界同步。完成的 step/reset 会消费输入并同步后端工作。

`DEVICE_RESIDENT` 中 `device=None` 表示后端的精确 CUDA 设备，返回的活跃 view 逻辑上只读。`HOST_BRIDGE` 中 `device=None` 表示 CPU；显式传入设备会请求 H2D 复制，返回值是 detached 快照。`TensorLifecycleCapabilities.state_fields` 是可机读的广义状态字段集合（`qpos`、`qvel`、`ctrl`、`time`，以及可用时的 `sensordata`）。body pose/velocity 消费方使用命名 sensor view，而不是引擎私有 body 数组约定。

- MJWarp 声明 `DEVICE_RESIDENT`：Torch 控制与选中 reset 行复制或 scatter 到稳定 MJWarp 设备存储，物理在 CUDA 执行，并通过 DLPack 暴露活跃 state view。`step_tensor()` 完成物理并将 tracked-sensor 刷新保持为 pending；首次读取 tracked tensor sensor 或 `sensordata` 时只刷新设备端状态，legacy NumPy generalized/body host cache 仅在混用回主机路径时惰性刷新。
- MJWarp tensor stepping 不支持 host pre-step callback；其最小选中行 reset 不支持模型随机化、fixed variants、待处理 interval wrench，以及带 mocap body 的模型。混用 legacy 写入会先刷新 host mirror，避免未选中 device 行回退。
- MuJoCo/MJBatch 声明 `HOST_BRIDGE`：加速器控制与 reset 行经过显式主机边界，CPU 物理仍是权威执行源，请求的 state 或 sensor 复制到选定 Torch 设备。其 tensor reset 复用既有 reset randomization 与 fixed-variant host 路径；tensor stepping 不支持 host pre-step callback。传输优化仍属适配器工作，不隐含设备端物理执行声明。
- 其余适配器保持 `UNSUPPORTED`，抛出 `NotImplementedError`，不会静默经 NumPy 转换。

## 相关决策

- [能力与证据 ADR](adr-capabilities.md) 定义快速失败声明与运行时证据。
- [实体与状态 ADR](adr-entities.md) 定义选中行 reset 与公开 state 布局边界。
- [Benchmark API 预留](benchmark-api.md) 定义稳定 benchmark 契约的 schema、同步、digest 与溯源要求。

## 影响

该契约保持既有 NumPy 生命周期兼容，并让 host-device 边界可测量。它还不是完整的 GPU collector/replay 协议：`uni_rl`、IPC、replay ingress 与 Manager-Based task dispatch 需要后续按版本迁移。后端调用方应在热路径值检查会强制标量设备同步的地方验证生产者 tensor；适配器继续校验元数据，并对不支持的生命周期特性快速失败。
