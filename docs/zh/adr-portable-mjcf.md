# 设计决策：可移植 MJCF 场景编译

[English](../en/adr-portable-mjcf.md) | [中文](adr-portable-mjcf.md)

## 状态、范围与所有者

**状态：已接受（profile v1）。** 本决策扩展并受限于[实体与不可变身份决策](adr-entities.md)和[能力与证据决策](adr-capabilities.md)。范围是冷路径 authoring、结构编译、canonical identity 与 source intent；原生物化、effective report 与执行仍归 adapter。

UniSim 拥有 portable profile、源/资源解析、canonical expanded MJCF、`CompiledSceneLayout`、assignment 语义和内容身份。MuJoCo `MjSpec` 只是解析受限 profile 的 structural oracle。Adapter 消费同一结果并通过公开引擎 API 翻译，不另建 composer 或身份方案。UniLab 继续拥有资产注册、任务配置、Manager 调度、policy I/O 与训练。

## 背景

M2 已公开实体、mirror、不可变 variant assignment 和 selected reset，但工作 composer 位于 MuJoCo adapter。Isaac worker 已经复用它；临时路径与 adapter 位置不是稳定的跨后端契约。[Roadmap #154](https://github.com/unilabsim/unisim/issues/154) 要求在缓存与 native 后端扩展前先收敛唯一可信 authoring 路径。

## 决策

Portable profile v1 以受限 MJCF 作为唯一物理 authoring source。`SceneCfg.entity_assets`、`mirror_of`、`entity_variant` 与 `FixedVariantPlan.assignment` 定义全部实体、mirror 和 variant 关系；源文件组织、body 顺序和外观不推断所有权。

公共冷路径：

1. 用 MuJoCo `MjSpec` 解析每个 entity source；
2. 在 attach 前拒绝 profile v1 之外的语义；
3. 为 entity 建立命名空间，按编译后的公共地址合并 keyframe，并要求 global option 兼容；
4. 每个 unique variant 独立编译；
5. 从最终 compiled model 反查 body、joint、geom、site、actuator、qpos 和 qvel 地址并冻结到 `CompiledSceneLayout`；
6. 输出序列化 expanded MJCF 供 adapter 物化；以及
7. 输出 source provenance、source/intent report 和版本化内容身份。

Structural oracle 采用 lazy 加载，使用 `scene-compiler` extra（`mujoco~=3.11.0`），不要求 MuJoCo adapter 的 `mjbatch` executor。导入 UniSim 或 SDK-free contract 模块不加载引擎。旧 whole-model `model_file` 入口和显式 native profile 仍是 adapter 路径，不构成 portable profile 支持声明。

### 受限 profile v1

每个 entity 有一个命名 root body，没有 world-body geometry 或 geom。除源 root 外 body 必须命名。floating entity 的根移动性是一个 root free joint；fixed entity 无 joint；kinematic mirror 与 kinematic rigid entity 使用编译器生成的 mocap root。非 root hinge、slide、ball joint 必须命名。Rigid entity 不能包含非 root joint；kinematic articulation 不支持。

Body geometry、显式 inertial、mesh、texture、hfield、命名 keyframe、contact 声明、sensor 与 joint-transmission actuator 只有在 structural oracle 接受其组合时才属于 profile。场景级 sensor fragment 只能包含上述有序 geom-pair contact 声明，或对象引用使用最终命名空间的 world-referenced `framepos`/`framequat` 声明。Tendon、equality、非 joint transmission、跨 entity 约束、源 `<include>`、inline asset override、非默认 compiler transform、不一致 global option、歧义 keyframe 名称/时间以及 variant topology/sensor 变化均 fail closed。Unsupported 或未验证的 native 语义绝不能被静默丢弃；adapter 必须拒绝，或在 effective report 中逐项记录显式近似。

Mirror 是无碰撞、无控制的视觉角色。它继承 source 与选中 variant 身份，绝不继承 target pose 或物理影响。

### 身份与报告

`SceneContentIdentity` schema version 1 哈希每个源的 entity 角色、kind、root/collision 角色、初始 pose、格式、mirror 关系、variant 角色与源字节，引用 mesh、texture、hfield 的逻辑路径与字节，以及 profile 身份、structural oracle 身份/版本、timestep、keyframe 选择和不可变 assignment。

它刻意排除绝对 checkout 路径与 adapter/runtime 设置。绝对位置保留在 provenance。USD cache owner 在 canonical identity 上追加 importer 参数与 Isaac/importer/runtime 版本，绝不替换它。Role bake 同样从 raw artifact identity 加上 collision、visual、mirror 与 bake 参数派生独立身份。

`SceneIntentReport` 可序列化，只包含 source intent 与 provenance，不能携带 native effective 值。物化后各 adapter 从实际 native readback 提供 `ImportReport` effective 字段，并限定 backend、runtime、profile、config 与生命周期。跨引擎数值相同不是验收标准。

## 备选方案

- **每个 adapter 自建 composer**：会重复身份与命名空间规则，使缓存正确性依赖后端，已拒绝。
- **URDF 作为 portable source**：v1 拒绝。URDF 保留 authoring vocabulary 或显式 translated/native compatibility profile；本 roadmap 需要的物理语义在 MJCF。
- **USD 作为真相源**：USD 是物化与缓存产物，不是物理 authoring contract。
- **通用资产 IR**：范围不必要。版本化 expanded MJCF、冻结公共 layout 与报告是最小完整桥梁。

## 验证边界

Focused tests 覆盖 SDK-free report/identity schema、路径迁移不变性、资源与 compiler 失效、下游 artifact identity 扩展、缺失 compiler 诊断、源 `<include>` 拒绝，以及 robot、passive object、table、mirror 与 N5 assignment `[1,1,0,1,0]` 的 golden 场景。它们不声明任何后端 native 支持。每个 adapter 必须物化公共结果、读回 effective identity/configuration，并通过自身 native 测试后才能扩大 support matrix。

Portable profile 的 `SceneCfg.fragment_files` 只能添加场景级、仅含 sensor 的 MJCF fragment，用于跨 entity `contact data="force" reduce="netforce"` 碰撞对力声明、`contact data="found" num="1"` found 声明，或使用最终限定名的 world-referenced body/site 位姿声明；对象引用使用最终 `entity/local-name` 命名空间。entity source 本身仍必须独立合法，compiler 在全部 entity attach 之后、每个 variant 编译之前解析 fragment，其他 fragment authoring 均 fail closed。每个 fragment 的源字节按声明顺序参与 canonical identity；focused tests 同时覆盖 cross-entity 解析、身份失效与 fail-closed 边界。
