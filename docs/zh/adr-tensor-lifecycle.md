# ADR：可选后端 Tensor 生命周期

[English](../en/adr-tensor-lifecycle.md) | [中文](adr-tensor-lifecycle.md)

## 状态

已接受（2026-09-27）

## 背景

UniLab 的 Manager-Based 运行时与 collector 目前以 NumPy 为边界，而 GPU 物理引擎已经暴露稳定的批量设备数组。若用各引擎私有 tensor API 直接全局替换 NumPy，会破坏既有 manager、replay 与 playback 消费者。CPU 物理引擎在 manager 计算加速后也仍需可用。

## 决策

`SimBackend` 暴露可选且默认快速失败的 tensor 生命周期，而不是全局替换环境契约。`TensorExecution` 区分 `DEVICE_RESIDENT`、`HOST_BRIDGE` 与默认 `UNSUPPORTED`；`get_tensor_capabilities()` 暴露各方法与 reset 特性的部分支持边界，同时声明进程拓扑、bulk 数据面、stream/event 所有权，以及 accepted Torch 设备。`UNSUPPORTED` 生命周期必须保持默认 in-process 拓扑与无数据面。有效的支持组合是进程内 direct device storage、进程内 host-bridge storage、设备驻留物理的外部 worker CUDA IPC，或 host-bridge 物理的外部 worker host shared memory；粗分类不隐含任何可选 tensor 方法已支持。

能力构造同样拒绝矛盾声明：`UNSUPPORTED` 不携带特性、字段、所有权或设备元数据；state views 必须有字段；selected reset 必须包含 `qpos` 与 `qvel`；reset randomization 必须依赖 selected reset；device reset randomization 必须依赖进程内 device-resident direct-storage 矩阵上的 reset randomization；packed plan 必须使用进程内 host-bridge 矩阵。

声明支持的适配器可实现 `get_state_views()`、`get_sensor_view()`、`step_tensor()` 与 `set_state_tensor()`。host-bridge 适配器还可声明 `packed_host_bridge`，并把 `TensorIOSpec` 编译为 `HostBridgeTransferPlan`。`set_state_tensor()` 接受主机 NumPy `ResetRandomizationPayload`；仅当适配器声明 `device_reset_randomization` 时，还接受 `TensorResetRandomizationPayload`——其引擎原生设备数组持有用于原地 scatter 的选中行最终绝对值；其余适配器对设备 payload 快速失败。

基础包仍不依赖 Torch；适配器按需 lazy-import tensor 运行时，并拥有 stream、布局与传输语义。

外部 GPU worker 保持 SDK 解释器与依赖隔离。pipe 只携带控制消息和错误；bulk tensor 通过稳定的 CUDA IPC arena 交叉。共享的 SDK-free transport 导出 opaque memory/event driver handle，并携带 ABI 版本、物理设备 UUID、大小与对齐。它绝不序列化 Torch-private storage、不在 host 进程 import Isaac SDK，也不把引擎原生 buffer 当作公开稳定指针。transport primitive 本身不代表 IsaacGym 或 IsaacSim 已具备 tensor 生命周期，其既有 CPU shared-memory subprocess 路径也不构成 tensor 支持声明。

Tensor 输入必须是连续 Torch tensor。control 是 float32 `(num_envs, num_actuators)`；reset row ID 是 `[0, num_envs)` 内唯一的 int64，qpos/qvel 是同一 accepted device 上的 float32 `(rows, nq/nv)`。生产方负责有限性检查，避免适配器在热路径强制标量同步；row 校验允许一次有界同步。完成的 step/reset 会消费输入并同步后端工作。

`DEVICE_RESIDENT` 中 `device=None` 表示后端的精确 CUDA 设备，返回的活跃 view 逻辑上只读。`HOST_BRIDGE` 中 `device=None` 表示 CPU；显式传入设备会请求 H2D 复制，返回值是 detached 快照。`TensorLifecycleCapabilities.state_fields` 是可机读的广义状态字段集合（`qpos`、`qvel`、`ctrl`、`time`，以及可用时的 `sensordata`）。body pose/velocity 消费方使用命名 sensor view，而不是引擎私有 body 数组约定。

packed host-bridge plan 在后端 materialize 后的冷路径编译。它冻结请求的 state/sensor 形状、offset、dtype、reset row ID 的 capacity/dtype/layout 与 body ID；预分配 host staging（仅 CUDA target 使用 pinned memory）和持久 Torch 目标；并分开暴露四个语义边界：CPU 物理前一次 packed control D2H、物理后一次 packed 全量 state/sensor H2D、一次选中行 reset D2H，以及一次 packed 选中行 post-reset H2D。空 reset 集不产生 reset 传输。选中行传输先复制连续 prefix，再在加速器上 scatter。操作携带逐边界 timing 以及累计方向、字节与同步计数。后端 close 会使 plan 失效；plan 是进程本地对象，不会随环境工厂序列化。

- MJWarp 声明进程内 direct storage 的 `DEVICE_RESIDENT`：Torch 控制与选中 reset 行复制或 scatter 到稳定 MJWarp 设备存储，物理在 CUDA 执行，并通过 DLPack 暴露活跃 state view。`step_tensor()` 完成物理并将 tracked-sensor 刷新保持为 pending；首次读取 tracked tensor sensor 或 `sensordata` 时只刷新设备端状态，legacy NumPy generalized/body host cache 仅在混用回主机路径时惰性刷新。
- MJWarp tensor stepping 不支持 host pre-step callback。选中行 reset 支持主机 NumPy randomization payload；对可设备 scatter 的子集（`body_mass`、`body_ipos`、`geom_friction` 与执行器 `kp`/`kd`），还支持设备驻留 `TensorResetRandomizationPayload`——通过逐世界 Model 行原地 scatter 写回，无主机往返；被 scatter 的行会把对应 host DR mirror 标记为过期，下一次主机读取时从设备刷新；质量/COM scatter 会在 reset graph 重放前触发一次即时的派生常量刷新。设备 payload 的其他字段、fixed variants、待处理 interval wrench，以及带 mocap body 的模型均快速失败。混用 legacy 写入会先刷新 host mirror，避免未选中 device 行回退。
- MuJoCo/MJBatch 声明进程内 `HOST_BRIDGE` 并实现 packed plan：加速器控制与 reset 行经过显式主机边界，CPU 物理仍是权威执行源，请求的 state 与 sensor 打包到一个稳定 H2D 布局并复制到选定 Torch 设备。其 tensor reset 把既有 NumPy reset-randomization payload 交给 host 适配器；packed stepping 不支持 host pre-step callback。这是传输布局优化，不是设备驻留物理声明。
- SuperDex 是已支持的 backend-owned packed `HOST_BRIDGE` tensor 后端，保持同样的四个语义边界。合成 tracked-body sensor 名称共享 packed state/sensor H2D 边界；作者声明的 accelerometer 采用 declared-but-unused 策略，在请求或绑定时快速失败。它接受 CPU/CUDA tensor；fixed variants、reset randomization 与 host pre-step callback 快速失败。支持提升基于 #1678 的 packed 边界计数审计、same-engine publication/control parity、full-public G1 parity、hidden-transfer profile、legacy path 检查与 2048 环境 runtime benchmark；这不表示与 MuJoCo 的 contact/solver 等价。
- MotrixSim 是已支持的 backend-owned packed `HOST_BRIDGE` tensor 后端，初始范围覆盖 no-variant whole-MJCF 与 portable profile。CPU 物理保持权威，接受 CPU/CUDA tensor，在冷路径冻结 public mapping 与 selected-row scratch，并保持每个语义边界最多一次传输。reset randomization、fixed variants 与 host pre-step callback 快速失败。支持提升基于 #1680 的 whole/portable lifecycle parity、hidden-host-copy 审计、quaternion/public-layout 测试与 idle 2048 环境 G1 benchmark。
- Drake 是已支持的 backend-owned packed `HOST_BRIDGE` tensor 后端，保持每个语义边界最多一次传输。它接受 CPU/CUDA tensor，CPU 物理保持权威，在 float32 packet 与权威 float64 Drake state 之间显式转换，将 public tracked-body view 纳入 packed state/sensor 边界，并对 fixed variants、reset randomization、host pre-step callback、各自不支持的 reset 工作，以及 pending interval body force 快速失败。支持提升基于 #1679 的 precision parity、selected-row isolation、packed counter 测试、native runtime 构建验证与 2048 环境 G1 benchmark。
- Newton 是窄条件已支持的进程内 direct `DEVICE_RESIDENT` tensor 后端，范围为单个 CUDA 绑定 articulation。它覆盖 `qpos`/`qvel`、callback-free tensor stepping、已审查的 non-portable selected-reset profile，以及从 Newton public link array 派生的协商 scalar/tracked-body sensor 子集；协商 scalar 集包含 `pelvis_local_linvel`、`torso_gyro` 与 `torso_upvector` frame-z-axis。CUDA graph/eager parity、hidden-transfer 审计、canonical G1 gross-threshold acceptance 与 2048 环境 G1 benchmark 构成支持提升证据。其他 sensor、portable multi-entity selected reset、reset randomization、fixed variants 与 host pre-step callback 快速失败。
- Genesis 是窄条件已支持的进程内 direct `DEVICE_RESIDENT` tensor 后端，范围为单个 non-portable CUDA 绑定 articulation，并要求 Genesis exact CUDA backend、zero-copy session 与 public Torch getter。它覆盖 `qpos`/`qvel`、callback-free tensor stepping、已审查的 non-portable selected reset，以及协商 scalar/tracked-body sensor 子集（`pelvis_local_linvel`、`torso_gyro` 与 `torso_upvector` frame-z-axis）。backend-owned 稳定 CUDA mirror 吸收 Genesis 1.3.3 public getter 的拷贝/非 contiguous 行为，且没有 bulk host detour。CPU/ROCm、portable 多实体、关闭 zero-copy、不支持 sensor、randomization、fixed variant 与 callback 路径快速失败。支持提升基于 canonical G1 gross-threshold parity、hidden-transfer attribution、2048 环境 benchmark 与独立物理 GPU acceptance；tracked-body 速度差异归类为 contact/solver divergence（MJWarp 对照组也可见），不构成 publication-frame 声明，也不是 portable Genesis 支持。
- IsaacGym 是窄条件已支持的 external-worker CUDA IPC tensor 后端，范围为已审查 GPU-pipeline profile。Preview 4 保留在 Python 3.8 worker，host 不 import IsaacGym；metadata-only worker command、same-GPU raw arena/event 与 D2D native projection 覆盖 `qpos`/`qvel`、callback-free stepping、selected reset，以及协商的 G1 scalar/tracked-body view。`pelvis_local_linvel` 包含 sensor-site point velocity；在第一个 tensor step 刷新 Isaac rigid-body state 前，reset 时 body/scalar 读取快速失败。reset randomization、fixed variant、callback、contact/body-wrench view、不支持的 sensor、CPU pipeline 与跨设备 IPC 均快速失败。支持提升基于 full-G1 与重复 mapped-generalized acceptance、hidden-transfer profile、fault/shutdown lifecycle 覆盖、同 GPU 双 worker 证据与 2048 环境 benchmark；这不表示 multi-GPU scaling 或任意 importer 支持。
- IsaacSim 是窄条件已支持的 external-worker CUDA IPC tensor 后端，范围为 opt-in mapped entity scene。Kit/IsaacLab 保留在专用 worker；same-GPU raw arena/event 为 device-resident `qpos`/`qvel`、callback-free stepping、mapped selected reset 与协商 scalar/tracked-body sensor view 取代 legacy CPU shared-memory bridge。worker command 只携带 metadata；reset randomization、fixed variants、host callback、不支持 sensor、legacy scene 与 CUDA 缺失均快速失败。公开 Manager surface 还包括 canonical `get_public_state_widths()`、由 inventory 支撑的 entity-qualified sensor/body 名称，以及聚合 `get_tracked_body_views()` block；后者使用 owner-qualified logical body name，因此重复 local body name 会快速失败。selected reset 声明 `AUTHORITATIVE_VIEWS`：`set_state_tensor()` 返回时 qpos/qvel、scalar sensor 与 tracked-body block 立即公开，无需 readiness physics step。native/reset/publication fault 会标记 worker faulted，后续 CUDA IPC command 快速失败。#350 证据新增 mapped multi-entity acceptance、full-G1 acceptance、mapped physics-state playback acceptance，以及 `scripts/benchmark/outputs/isaacsim-issue350/` 下的 8/2048 环境 phase-local regression artifact。支持提升基于 full-G1 large-finite-box acceptance、mapped multi-entity generalized acceptance、hidden-transfer audit、fault/shutdown lifecycle 覆盖、2048 环境 benchmark 与同 GPU 双 worker 并发证据。这不表示 contact sensor、infinite plane 或生产 multi-GPU scaling 支持。
- 每个未审查的适配器或运行时 profile 都在公开 support matrix 中保持 tensor `UNSUPPORTED`，抛出 `NotImplementedError`，不会静默经 NumPy 转换。特别是 IsaacGym 的既有 CPU shared-memory subprocess 路径不构成 tensor 支持声明。

## 相关决策

- [能力与证据 ADR](adr-capabilities.md) 定义快速失败声明与运行时证据。
- [实体与状态 ADR](adr-entities.md) 定义选中行 reset 与公开 state 布局边界。
- [Benchmark API 预留](benchmark-api.md) 定义稳定 benchmark 契约的 schema、同步、digest 与溯源要求。

## 影响

该契约保持既有 NumPy 生命周期兼容，并让 host-device 边界可测量。它还不是完整的 GPU collector/replay 协议：`uni_rl`、IPC、replay ingress 与 Manager-Based task dispatch 需要后续按版本迁移。后端调用方应在热路径值检查会强制标量设备同步的地方验证生产者 tensor；适配器继续校验元数据，并对不支持的生命周期特性快速失败。
