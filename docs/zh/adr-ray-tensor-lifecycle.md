# ADR：Ray Tensor 生命周期

[English](../en/adr-ray-tensor-lifecycle.md) | [中文](adr-ray-tensor-lifecycle.md)

## 状态

已接受，适用于 2.x tensor ray contract（#462，首个实现切片）。[#472](adr-ray-perception-queries.md) 已补充 perception geometry、filter、minimum distance 与可选 tensor normal。

## 背景

Phase-1 的 `uni_ray` 通过 host NumPy pose bridge 建立了正确性：设备端位姿被下载、转换、上传，执行 trace 后再读回，随后 observation 往往还要重新上传。该数据面与 UniSim 的 tensor lifecycle 冲突。`DEVICE_RESIDENT` 后端已经公开权威 tracked-body Torch view；`HOST_BRIDGE` 后端已经把一个 packed state/sensor packet 发布到 Torch collector；UniLab observation 是 Torch-native 的，并可选择 CPU 或 CUDA carrier。

仅为“过一次 host 再回来”的 ray 查询还引入了排序风险：生产者 kernel、Warp kernel 与消费者 Torch kernel 可能位于不同 stream，而 caster 持有的可复用输出也可能在调用方读取前被覆盖。公开指针、Warp array、引擎 data object、CUDA stream 或 event 只会把复杂性推进任务代码，并让 UniSim 耦合到某个 caster 实现。

## 决策

ray-query 插件契约保持两个 profile，并在边界上保持 backend-private。既有 NumPy profile 是工具、调试与一致性验证 surface；训练 collector 使用 tensor profile：

```text
bind_pose_layout(body_names)
  → update_pose_tensor(RayPoseTensorView)
  → trace_tensor(...)
  → RayTraceTensorResult
```

公开 tensor 类型为 `RayPoseTensorView`、`RayTraceTensorResult`，以及 `RayCasterCapabilities` 的 tensor 字段。Torch 仍是惰性运行时依赖：导入 `unisim` 或 caster 包不得 eager-import Torch、Warp 或引擎 SDK。`MjModel`、`MjData`、`mjwarp.Data`、`SceneData`、原生 model、Warp array、指针、stream 或 event 都不得跨过边界。

Pose source 归调用方所有。调用方传入按 `body_names` 命名的全 batch float32 Torch view：position 形状为 `(E, B, 3)`，单位四元数形状为 `(E, B, 4)`，使用公开 `wxyz` 顺序。完全 contiguous 的存储有效；packed packet 列 view 在最后一维 stride 为 1、且所有 stride 为正、不重叠时同样有效。caster 仅在一次 update 的生命周期内 alias 该存储；它不复制、不重排、不转置、不静默调用 `.contiguous()`、不延长 source 生命周期，也不把该 tensor 保留为下一次 source。bind 阶段一次性解析不可变名称布局。第二次 bind、body 缺失、顺序错误、设备错误、dtype 错误或不支持的存储模式都 fail closed。

`DEVICE_RESIDENT` adapter（本切片为 MJWarp）通过公开 tracked-body Torch view 输入 caster。ray 热路径 pose D2H/H2D 均为 0，也没有 result 经由 host 的绕行。`HOST_BRIDGE` adapter（MuJoCo 与 MotrixSim）把请求的 `body_pos_w` 与 `body_quat_w` 作为 contiguous column block 加入既有 packed packet。该 packet 仍是唯一公开 H2D：state、sensor 与 ray pose block 一起传输，没有 ray 专属传输或 NumPy 转换。CPU collector 使用对应 CPU packet，不额外制造 host-device crossing。

Torch/Warp 边界是由 caster 构造的 zero-copy 存储 alias。公开四元数存储保持 `wxyz`；如 caster 私有存储需要组件顺序（例如 `xyzw`），由 Warp kernel 在 scatter 到 caster 自有 buffer 时完成转换。vec 类型 pointer alias 不得拥有或释放调用方存储。不支持的 dtype、shape、device 或 stride 组合 fail closed，而不是通过隐藏复制归一化。

Stream 与 event ownership 为 `caller`。kernel 启动在调用方当前 stream 上；caster 不创建公开同步原语，也不在 tensor 路径同步设备。调用方后续操作必须在读取 persistent output 前保持该 stream 顺序。显式 host 校验只能出现在已记录的冷路径；若热路径数值检查会强制标量设备读取，则延迟到 device-side reduction 或调用方显式选择的 debug read gate。

输出是 caster 持有的 persistent buffer。`trace_tensor` 按 batch shape 与请求输出 profile 复用 tensor，在设备端计算可选 hit point 与 body ID，并返回 view。后续 trace 可能覆盖这些值。调用方若需要跨调用保留结果，必须写入自己的 carrier。重建或场景替换是冷路径：关闭 caster，获取新的 `RaySceneDescription`，materialize 新 caster，并重新 bind pose layout。既有 plan 或 caster view 在 close 后失效；selected-row publication 只写选中的 packet 行。

### 后端场景投影

后端通过 `SimBackend.get_ray_scene_description(body_names=None)` 公开 collision geometry。detached descriptor 保持公开 geom 顺序，把 geom body ID remap 到选中的公开 body 轴，并为 MuJoCo family 中匿名 body 合成稳定的 `body_<i>` alias，而不改写原生 model 名称。未知、重复、为空或无法解析的 body 名称 fail closed。该投影是元数据，永远不会在 ray 热路径解析或刷新。

## 结果

MJWarp 可以端到端保持 GPU-resident；MuJoCo 与 MotrixSim 复用一个可度量的 packed H2D 边界。该契约让传输、同步、分配与 stream 顺序行为可审计，而不是依赖隐式复制。

caster 必须实现严格的元数据校验与 caller-owned storage 纪律；调用方若要保留可复用输出，也必须自行复制。设备端错误需要 device reduction 或显式 debug read gate 才能成为 Python 诊断。初始 tensor result 刻意不包含 normal；mesh 支持仍是 caster capability 决策，而不是通用后端承诺。这些限制换来的是没有隐藏 host round trip、且没有 backend-private 泄漏的热路径。

#462 排除 normal 与 public mesh topology 是初始切片边界，不是永久 contract 决策。#472 新增 detached collision companion 与单独协商的 tensor normal 输出，同时保留本 ADR 的全部所有权与 stream 规则。

## 相关决策

- [Tensor lifecycle ADR](../en/adr-tensor-lifecycle.md)定义后端 tensor 执行与 packed host-bridge plan。
- [Packed reset lifecycle ADR](../en/adr-packed-reset-lifecycle.md)定义 selected-row publication 与 plan transaction ownership。
- [Ray-query contract](../zh/ray-query.md)记录具体公开 API 与 capability 矩阵。
- [Perception-ready ray geometry ADR](adr-ray-perception-queries.md)定义 public collision companion、static filter、minimum distance 与 optional normal。
