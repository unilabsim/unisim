# ADR：统一 Genesis device-resident DR 生命周期

[English](../en/adr-genesis-dr-lifecycle.md) | [中文](adr-genesis-dr-lifecycle.md)

## 状态

已在 1.x 契约线接受（UniLab issue #2520）。

## 背景

此前 Genesis 的域随机化按场景 materializer 分裂：已审查的 whole-MJCF CUDA lane 暴露 device-resident reset 字段，而 portable `entity_assets` 场景携带 host reset setter 与仅 portable 的 pending wrench 路径。该分裂把较宽的 portable reset 集合（`base_mass_delta`、COM、DOF 与 actuator 字段）声明为支持，使热路径 operand 经过 host 数组，并允许 reset/interval 行为依赖场景模式。它还让 CPU session 看起来支持 runtime DR，尽管这些 host setter 没有经过审查的完整生命周期；聚合 capability report 也没有投射 device-only term 集合。

Portable materializer 已经拥有 device 路径所需的不可变公开到原生映射：entity body ID、原生 link index、actuator 列以及按 variant 冻结的冷默认值。Genesis 1.3.3 在 zero-copy CUDA/AMDGPU lane 上为选中行质量/增益提供公共 Torch setter，并为 solver wrench 提供公共提交 API。

## 决策

Genesis 只有同一条 runtime DR 生命周期，且它是 device-resident 的：

- Whole-MJCF 与 portable 场景均使用 exact `gs.cuda` + 非 HIP Torch CUDA，或 exact `gs.amdgpu` + HIP/ROCm Torch，配合 Genesis zero-copy 与公共 Torch lane。CPU、Metal、runtime family 不匹配以及关闭 zero-copy 的 session 不声明 runtime DR。
- Reset randomization 是选中行最终值子集 `body_mass`、actuator `kp` 与 actuator `kd`。校验与提交保持在已接受 CUDA device 上的 Torch tensor 中。Portable 行使用缓存的 device-resident 公开/原生 index 表，并通过审计后的所属原生实体 scatter。Host/CPU reset payload 按项与平面点名并快速失败。
- Interval randomization 是世界系 `body_force` 与 `body_torque` 子集。Torch int64 且经过审计的公开 body ID 选取缓存原生 solver link；float32 CUDA force/torque operand 校验形状、dtype、device、有限性与物理归属后，在每个 link COM 为下一个原生 step 提交。未声明的 device 平面 interval 项按名称快速失败。
- 新 interval plan 会在自身 op 累积前取消未消费 wrench。直接提交与同一 plan 内多个 op 独立加性累积。选中 tensor 或 entity reset 会在 Genesis 原生状态 setter 清空 solver staging 之前取消匹配行/body；无关行与 body 保留。成功 step 会消费并清空 pending mirror。
- 冷边界显式限定为启动、场景物化、variant 物化、不可变公开/原生 index 构建与冷默认表。热 reset 与 interval 执行不做 host payload 绕行。
- Legacy portable host reset/interval 路径及其 capability 广告被直接删除，没有 alias 或基于时间的兼容名。Portable host-only setter 不是 runtime DR，也不被声明。NumPy 只保留在冷元数据/序列化 leaf。
- `get_dr_capabilities()` 是权威来源。没有 device 生命周期时（也就是所有 session 的 host 平面）term 集合为空。激活 session 只填充 `supported_device_reset_terms` 与 `supported_device_interval_terms`，host 集合保持为空。聚合 `get_capabilities()` 将两个平面独立投射为 `dr.device.reset.*` 与 `dr.device.interval.*`；既有 `dr.reset.*` 与 `dr.interval.*` key 继续只描述 host 平面。
- Portable reset 集合随之收窄：COM、base delta、DOF friction/damping/armature、geometry、gravity 与其他 reset 项在两个平面上均不支持，直到存在经过审查的 device 生命周期与验收证据。

## 证据

- 真实 CUDA 验收覆盖 whole-MJCF 与 portable 场景，位于 `tests/adapters/genesis/test_tensor_runtime.py`。portable 测试物化真实多 entity/variant 场景，检查空的 host 集合、填充的 device 集合以及分开的聚合 key 家族，通过 selected-row 与 mask reset 只修改选中行，验证直接提交与 plan 驱动的 wrench staging 及 step 消费/清空，并在已接受 device 上覆盖形状、有限性、行/body 范围、归属、dtype 与非 Torch 校验失败。
- 单独以 `CUDA_VISIBLE_DEVICES=""` 屏蔽 CPU 进程，验证 portable capability 与 tensor-DR 声明为空，且 CPU/host 请求快速失败。被跳过的测试不构成验收证据。

## 影响

- 消费方不能把 portable Genesis DR 移到 CPU 或 torch.cpu operand。owner 必须在已接受 CUDA device 上采样，并使用 torch-first 公开 payload。
- Portable 场景拥有更小但一致的 reset 契约。被删除的 host setter 不提供兼容路径；迁移目标是 device 生命周期或其他适配器。
- 聚合支持来自实例 API，而不是静态场景模式叙述，因此 CPU session 不再继承 portable DR 声明。
- Pending wrench 在 reset 与 plan 替换时确定性地取消，普通 step 仍是消费边界。部分原生提交/取消失败会使 backend fault。

## 相关决策

- [tensor 生命周期 ADR](adr-tensor-lifecycle.md) 定义 exact CUDA/AMDGPU 执行 lane。
- [torch-first DR payload ADR](adr-dr-torch-first-payloads.md) 定义 payload 家族与按平面的 capability 语义。
- [packed reset 生命周期 ADR](adr-packed-reset-lifecycle.md) 定义组合 reset 字段的事务。
- [实体与状态 ADR](adr-entities.md) 定义选中行公开状态语义。
