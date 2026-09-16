# 场景布局与 reset 映射

[English](../en/scene-layout.md) | [中文](scene-layout.md)

这是 [#108](https://github.com/unilabsim/unisim/issues/108) 下 [#109](https://github.com/unilabsim/unisim/issues/109) 的映射基础。[实体设计决策](adr-m2-entities.md) 定义声明和坐标系契约。布局类型与协议 helper 本身不代表 adapter 的多实体 runtime 已开放。

## 公共与原生地址

`CompiledSceneLayout` 包含 `EntityLayout`：实体局部 body/parent 名称、公共全局 body ID、root qpos/qvel 列、非根 `JointLayout` 和 actuator 名称/目标/control 列。浮动根占七个位置和六个速度列，固定/kinematic 根不占广义状态列。球关节占四个位置和三个速度列。被动关节有状态列，但不会隐式增加 actuator。

全场景广义位置、速度和控制列各自必须恰好覆盖一次，禁止空洞、重复或越界。Body ID 唯一且有界；引擎 world body 可不归任何实体。原生 actor handle、tensor offset 和资产身份由 adapter 单独保存。实体 body 顺序不必是拓扑顺序，地址也不必连续。

具名查询要求 `entity/local_name` 或显式 `entity` 参数。`require_same_layout` 比较完整名称、parent 拓扑、joint 类型、root 模式、actuator 目标、顺序与地址，而不只比较维度。Adapter 可先将原生数据重映射为此公共顺序，但不能把不同公共签名当作相同布局。

## Reset 准备

`layout.validate_reset(request, num_envs=...)` 校验所有 patch 并返回完整 `BoundSceneReset`，不 yield 部分计划或写状态。在调用方能提交原生写入前，已检查 root 模式权限、joint 选择、打包宽度和球关节四元数。固定根拒绝位姿/速度写，kinematic 根只允许位姿写。

`prepare_scene_reset` 消费一致的广义状态/root 快照，构造独立的选中行和显式写 mask。缺失字段保留当前值。浮动根角速度在公共世界系和广义状态 body 系间转换；只修改姿态时，通过调整广义角速度表示来保持世界系角速度。Kinematic 位姿独立于广义状态。包括校验失败在内，输入快照均不被修改。

Adapter 仍负责原生索引映射、选中控制/wrench 清理、刷新和故障处理。这些 helper 不实现原生回滚。Adapter 尚未发布物化布局时，`SimBackend.get_scene_layout()` 明确拒绝。

## Worker 边界

Scene wire schema version 1 独立于 M1 配置报告版本。`to_dict`/`from_dict` 在每层检查精确字段集合、版本及完整布局有效性。同一模块可在 Python 3.8 worker 按文件路径加载，只依赖标准库和 NumPy；仅 host 校验请求时局部导入宿主 reset 请求类型。

映射槽位区分 `(N, nq)`、`(N, nv)`、`(N, nu)` 和 `(N, E, 13)` 实体根。Reset mask 分别标识位置、速度和根通道。Worker 在 attach 任何内存前检查全部槽名、shape 与 dtype。零宽状态/动作槽保留零个公共元素，同时分配操作系统要求的最小非零共享内存。现有 worker 槽位在执行路径迁移前保持原 wire shape；这不建立第二套永久场景 runtime。

## 验证

契约测试覆盖独立根、被动关节、非连续索引、拓扑与 wire 篡改、局部 reset 校验及独立快照。旋转期望使用明确的非单位姿态和独立数值向量。真实 IsaacGym Python 3.8 解释器也已验证不导入 UniSim 即可加载共用校验器。原生场景执行、实例身份、reset 隔离与物理行为仍需 adapter 工作包验收；这些测试不能替代它们。
