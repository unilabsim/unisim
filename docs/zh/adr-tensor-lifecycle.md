# ADR：可选后端 Tensor 生命周期

[English](../en/adr-tensor-lifecycle.md) | [中文](adr-tensor-lifecycle.md)

## 状态

已接受（2026-09-27）

## 背景

UniLab 的 Manager-Based 运行时与 collector 目前以 NumPy 为边界，而 GPU 物理引擎已经暴露稳定的批量设备数组。若用各引擎私有 tensor API 直接全局替换 NumPy，会破坏既有 manager、replay 与 playback 消费者。CPU 物理引擎在 manager 计算加速后也仍需可用。

## 决策

`SimBackend` 暴露可选且默认快速失败的 tensor 生命周期，而不是全局替换环境契约。`TensorExecution` 区分 `DEVICE_RESIDENT`、`HOST_BRIDGE` 与默认 `UNSUPPORTED`；`get_tensor_capabilities()` 暴露各方法与 reset 特性的部分支持边界，同时声明进程拓扑、bulk 数据面、stream/event 所有权，以及 accepted Torch 设备。`UNSUPPORTED` 生命周期必须保持默认 in-process 拓扑与无数据面。有效的支持组合是进程内 direct device storage、进程内 host-bridge storage、设备驻留物理的外部 worker CUDA IPC，或 host-bridge 物理的外部 worker host shared memory；粗分类不隐含任何可选 tensor 方法已支持。

能力构造同样拒绝矛盾声明：`UNSUPPORTED` 不携带特性、字段、所有权或设备元数据；state views 必须有字段；selected reset 必须包含 `qpos` 与 `qvel`；reset randomization 必须依赖 selected reset；packed plan 必须使用进程内 host-bridge 矩阵。

声明支持的适配器可实现 `get_state_views()`、`get_sensor_view()`、`step_tensor()` 与 `set_state_tensor()`。host-bridge 适配器还可声明 `packed_host_bridge`，并把 `TensorIOSpec` 编译为 `HostBridgeTransferPlan`。

基础包仍不依赖 Torch；适配器按需 lazy-import tensor 运行时，并拥有 stream、布局与传输语义。

外部 GPU worker 保持 SDK 解释器与依赖隔离。pipe 只携带控制消息和错误；bulk tensor 通过稳定的 CUDA IPC arena 交叉。共享的 SDK-free transport 导出 opaque memory/event driver handle，并携带 ABI 版本、物理设备 UUID、大小与对齐。它绝不序列化 Torch-private storage、不在 host 进程 import Isaac SDK，也不把引擎原生 buffer 当作公开稳定指针。transport primitive 本身不代表 IsaacGym 或 IsaacSim 已具备 tensor 生命周期，其既有 CPU shared-memory subprocess 路径也不构成 tensor 支持声明。

Tensor 输入必须是连续 Torch tensor。control 是 float32 `(num_envs, num_actuators)`；reset row ID 是 `[0, num_envs)` 内唯一的 int64，qpos/qvel 是同一 accepted device 上的 float32 `(rows, nq/nv)`。生产方负责有限性检查，避免适配器在热路径强制标量同步；row 校验允许一次有界同步。完成的 step/reset 会消费输入并同步后端工作。

`DEVICE_RESIDENT` 中 `device=None` 表示后端的精确 CUDA 设备，返回的活跃 view 逻辑上只读。`HOST_BRIDGE` 中 `device=None` 表示 CPU；显式传入设备会请求 H2D 复制，返回值是 detached 快照。`TensorLifecycleCapabilities.state_fields` 是可机读的广义状态字段集合（`qpos`、`qvel`、`ctrl`、`time`，以及可用时的 `sensordata`）。body pose/velocity 消费方使用命名 sensor view，而不是引擎私有 body 数组约定。

packed host-bridge plan 在后端 materialize 后的冷路径编译。它冻结请求的 state/sensor 形状、offset、dtype、row ID 与 body ID；预分配 pinned host staging 和持久加速器目标；并分开暴露四个语义边界：CPU 物理前一次 packed control D2H、物理后一次 packed 全量 state/sensor H2D、一次选中行 reset D2H，以及一次 packed 选中行 post-reset H2D。空 reset 集不产生 reset 传输。选中行传输先复制连续 prefix，再在加速器上 scatter。操作携带逐边界 timing 以及累计方向、字节与同步计数。后端 close 会使 plan 失效；plan 是进程本地对象，不会随环境工厂序列化。

- MJWarp 声明进程内 direct storage 的 `DEVICE_RESIDENT`：Torch 控制与选中 reset 行复制或 scatter 到稳定 MJWarp 设备存储，物理在 CUDA 执行，并通过 DLPack 暴露活跃 state view。`step_tensor()` 完成物理并将 tracked-sensor 刷新保持为 pending；首次读取 tracked tensor sensor 或 `sensordata` 时只刷新设备端状态，legacy NumPy generalized/body host cache 仅在混用回主机路径时惰性刷新。
- MJWarp tensor stepping 不支持 host pre-step callback；其最小选中行 reset 不支持模型随机化、fixed variants、待处理 interval wrench，以及带 mocap body 的模型。混用 legacy 写入会先刷新 host mirror，避免未选中 device 行回退。
- MuJoCo/MJBatch 声明进程内 `HOST_BRIDGE` 并实现 packed plan：加速器控制与 reset 行经过显式主机边界，CPU 物理仍是权威执行源，请求的 state 与 sensor 打包到一个稳定 H2D 布局并复制到选定 Torch 设备。其 tensor reset 把既有 NumPy reset-randomization payload 交给 host 适配器；packed stepping 不支持 host pre-step callback。这是传输布局优化，不是设备驻留物理声明。
- SuperDex 拥有 backend-owned packed `HOST_BRIDGE` 实现，保持同样的四个语义边界。它接受 CPU/CUDA tensor，但 fixed variants、reset randomization 与 host pre-step callback 快速失败。在记录 generalized-task parity 与运行时 benchmark 证据前，静态 support matrix 不把它提升为 supported tensor backend。
- MotrixSim 与 Drake 也拥有 backend-owned packed `HOST_BRIDGE` 候选实现，保持每个语义边界最多一次传输。它们接受 CPU/CUDA tensor，CPU 物理保持权威，并对 fixed variants、reset randomization、host pre-step callback 以及各自不支持的 reset 工作快速失败；Drake 还会在 interval body force pending 时快速失败。这些实例候选实现不是静态支持提升；Drake 仍需要 native batch runtime 证据。
- Newton 拥有部分进程内 direct `DEVICE_RESIDENT` 生命周期，覆盖 `qpos`/`qvel`、tensor stepping 与当前 non-portable selected-reset profile。sensor 与 portable multi-entity selected reset 快速失败。在 sensor、parity 与 benchmark 证据完成前不做支持提升。
- Genesis 对 non-portable 单 articulation、exact CUDA backend、zero-copy session 与 public Torch getter 的窄条件路径拥有部分进程内 direct `DEVICE_RESIDENT` 候选。backend-owned 稳定 CUDA mirror 吸收 Genesis 1.3.3 public getter 的拷贝/非 contiguous 行为，且没有 bulk host detour；finite 责任归 producer，selected reset 将 row range 与 uniqueness 检查打包为一次 bounded scalar read。CPU/ROCm、portable 多实体、关闭 zero-copy、sensor、randomization、fixed variant 与 callback 路径快速失败。该候选不是 portable Genesis 支持提升，仍需要 sensor、parity、隐藏同步、benchmark 与多 GPU 证据。
- IsaacGym 具有窄条件实验性 external-worker CUDA IPC 实例候选，覆盖 `qpos`/`qvel` control、stepping 与 selected reset。SDK 保留在 Python 3.8 worker 中，host 进程不 import IsaacGym；worker pipe 只传控制 metadata 与错误，same-GPU raw memory/event handle 发布稳定 device arena。sensor、body/contact view、随机化、fixed variant、callback、完整 collector lifecycle parity 与 benchmark 在证据完成前均保持不支持。
- IsaacSim 具有窄条件 external-worker CUDA IPC 候选，覆盖 device-resident `qpos`/`qvel` control、stepping 与 selected reset。Kit/IsaacLab SDK 保留在 worker 中；opt-in tensor path 用 same-GPU raw arena/event 取代 legacy CPU shared-memory bridge。最小 real mapped-worker selected-reset parity 已通过，但 sensor、随机化、callback、generalized-task/G1 parity、fault/shutdown 覆盖、benchmark 与 promotion review 仍未完成。
- 其他适配器在公开 support matrix 中保持 tensor `UNSUPPORTED`，抛出 `NotImplementedError`，不会静默经 NumPy 转换。特别是 IsaacGym 与 IsaacSim 的既有 CPU shared-memory subprocess 路径不构成 tensor 支持声明。

## 相关决策

- [能力与证据 ADR](adr-capabilities.md) 定义快速失败声明与运行时证据。
- [实体与状态 ADR](adr-entities.md) 定义选中行 reset 与公开 state 布局边界。
- [Benchmark API 预留](benchmark-api.md) 定义稳定 benchmark 契约的 schema、同步、digest 与溯源要求。

## 影响

该契约保持既有 NumPy 生命周期兼容，并让 host-device 边界可测量。它还不是完整的 GPU collector/replay 协议：`uni_rl`、IPC、replay ingress 与 Manager-Based task dispatch 需要后续按版本迁移。后端调用方应在热路径值检查会强制标量设备同步的地方验证生产者 tensor；适配器继续校验元数据，并对不支持的生命周期特性快速失败。
