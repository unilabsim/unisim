# M2 设计决策：实体、固定身份与局部重置

[English](../en/adr-m2-entities.md) | [中文](adr-m2-entities.md)

## 状态、范围与 owner

**状态：Proposed for implementation，属于已获执行授权的 [#108](https://github.com/unilabsim/unisim/issues/108)。** 本文记录工作包 A 采用的公共声明，不表示运行时工作包 B–F 或任何新的后端组合 profile 已完成。[#84](https://github.com/unilabsim/unisim/issues/84) 是设计 owner；[#91 M2](https://github.com/unilabsim/unisim/issues/91) 是父里程碑。[#72](https://github.com/unilabsim/unisim/issues/72) 负责消费这些能力的操作场景；[#79](https://github.com/unilabsim/unisim/pull/79) 提供实现参考，不等于最终公共契约。

UniSim 负责物理实体声明、物化、原生映射、IPC、adapter 生命周期和一致性验收。UniLab 负责提交前的资产注册与物化、逻辑 selector、任务配置、Manager-Based 调度、策略 I/O 和 checkpoint 兼容。公共值不携带 SDK 对象，也不引入 UniLab 依赖。边界遵循 [ADR-0007](https://github.com/Motphys/UniLab/blob/main/docs/sphinx/source/adr/ADR-0007-unisim-extraction-boundary.md) 和 [ADR-0006](https://github.com/Motphys/UniLab/blob/main/docs/sphinx/source/adr/ADR-0006-community-manager-api-on-numpy-runtime.md)；能力与证据规则遵循 [M1 决策](adr-m1-capabilities.md)。

## 背景

机器人、可移动物体、静态桌面和视觉目标即使处于同一环境，也需要独立的身份和状态。物体可能含有贡献状态但不增加动作维度的被动关节。选择工具变体不应要求调用方为每种工具重复编写机器人和桌面的完整场景。actor 创建顺序以及 qpos/qvel 开头只有一个 root 的假设无法可靠表达这些需求。

设计分离源声明、编译后的公共布局和原生执行。当前切片实现源声明、reset 请求值，以及对不支持组合的拒绝。冻结原生布局、运行时执行及其证据仍属于后续实现工作。

## 实体与 variant 声明

公共值位于 `unisim.entities`；`SceneCfg` 承载它们在场景中的关系。

| 值 | 决策 |
| --- | --- |
| `EntityInitialState` | root link 位置和单位 `wxyz` 四元数；关节默认值由源/keyframe 持有，初始 root 速度为零。 |
| `SceneEntitySpec` | 稳定名称、`ModelSourceDescriptor`、显式格式、`articulation`/`rigid` 类型、`fixed`/`floating`/`kinematic` root 模式、初始位姿、碰撞标志和可选 `mirror_of`。 |
| `EntityVariantBinding` | 一个 `target_entity` 和一个既有的不可变 `FixedVariantPlan`；这是唯一的 variant consumer 绑定。 |
| `SceneCfg.entity_assets` | 物理源或镜像声明的 tuple；区别于仍表示逻辑 selector 映射的 `SceneCfg.entities`。 |
| `SceneCfg.entity_variant` | 可选绑定；首版最多一个物理 variant consumer，可有多个显式镜像。 |

实体名符合 `[A-Za-z][A-Za-z0-9_-]*`，在场景内唯一。物理实体必须有 source；descriptor 保存文件路径，不保存活跃引擎对象。可识别的格式名（`mjcf`、`urdf`、`usd`、`superdex_bot`）是声明词汇，不表示所有 adapter 都能导入。源拓扑、root 模式兼容性和格式组合支持需要 adapter 在物化时校验。

镜像直接引用已有物理实体：拒绝自引用、镜像链和不存在的目标。镜像继承源和 fixed variant identity，**不继承 pose**。它有自己的初始位姿，必须为 rigid、kinematic、禁用碰撞，不引入控制或物理影响。它不能声明另一份源，也不能作为 variant consumer。其声明格式必须与目标一致。因此，视觉目标可以在不同位置显示所选工具，而不改变被仿真的工具。

`FixedVariantPlan.assignment` 仍是最终、不可变的环境到 catalog 映射；已知 `num_envs` 时检查其长度。在实体绑定中，catalog sources 描述目标实体；在既有 `fixed_variant_plan` 入口中仍保留完整模型语义。Reset 不重新抽取身份。首版实现不能静默用 round-robin 替换显式 assignment。`same_layout` 要求语义布局相同，不只是数组等长；`uniform_public_layout` 需要 adapter 完成 remap、padding 和真实验证后才能声明支持。多个独立 consumer、reset 换拓扑和跨实体传动/约束需要独立决策。

## 坐标系与局部 reset 契约

Root pose 指向 **root link 原点**，不是质心。位置使用移除后端 clone 平移偏置后的环境世界系；姿态为 `wxyz` 顺序的单位四元数。Root velocity 先是 link 原点线速度，再是角速度，二者均在该世界系表达。原生 COM 速度或 body-frame 角速度必须由 adapter 转换。初始位姿遵循相同约定。

`EntityStatePatch` 寻址一个实体，可以携带下列字段的任意非空组合：

| 字段 | 列与含义 |
| --- | --- |
| `root_pose` | 七列：link 原点 xyz，然后是单位 wxyz。 |
| `root_velocity` | 六列：link 原点世界系线速度，然后是世界系角速度。 |
| `joint_positions` | 按 `joint_names` 顺序拼接 qpos 列，使用每个所选关节绑定后的 qpos 宽度。 |
| `joint_velocities` | 按 `joint_names` 顺序拼接 qvel 列，使用每个所选关节绑定后的 qvel 宽度。 |
| `joint_names` | 唯一的实体局部名称；空 tuple 选择绑定实体顺序中的全部非 root 关节。 |

不假定关节宽度为一：球关节的位置和速度宽度不同。Adapter 在写入前根据冻结布局检查名称、宽度、球关节四元数和 root 操作权限。Patch 能表达 root pose 不意味着固定 root 可以被写入。缺失字段表示保留，不表示清零。数值数组会复制到不可变存储，必须有限且为二维，各已提供字段行数相同。Root pose 四元数经过校验，不静默归一化调用方输入。

`SceneResetRequest(env_ids, patches)` 保留调用方选择的行顺序。环境 ID 为非空 tuple，元素为互不重复的非负整数，拒绝布尔 ID。每个 patch 的行数必须与其一致。一个实体只能有一个 patch，因此调用方须合并该实体的写入，不能提交相互冲突的重复 patch。运行时还要检查环境上界和实体/布局权限。

运行时事务必须包含四个阶段：

1. 在第一次原生写入前校验**所有** selector、patch、shape、frame 和值。校验失败保证状态零修改。
2. 通过冻结映射转换，只提交选中的环境、实体和字段。
3. 对受影响状态执行 M0 的控制目标、pending wrench 和缓存生命周期，再刷新。保留无关实体/环境状态和 fixed variant identity。
4. 原生提交中途失败时，要么执行经过证明的回滚，要么将 backend 标记为 faulted 并要求重建。不能继续 step，也不能把部分写入状态作为成功结果发布。

请求类型提供无需引擎即可进行的值校验；它们本身不实现 GPU 原子性、回滚、原生事务或 backend fault 状态。Adapter 执行层必须交付并测试这些保证。

`SimBackend` 声明 `get_entity_names()`、`get_entity_state(entity)` 和 `reset_entities(request)`；基类实现均显式抛出 `NotImplementedError`。已实现的状态读取必须按上述 root/joint 字段名返回独立数组，关节采用冻结的实体顺序，新鲜度遵循声明 profile；不可用状态必须报错，不能伪装为当前值。这些方法确立后续 adapter 接口，不表示执行已经实现。

## 编译布局与迁移

冷路径物化必须冻结实体限定的公共名称，以及公共到原生的 root/body/joint/actuator 映射。状态包含被动关节，动作只包含已声明 actuator。原生 actor/prim/body/DoF 索引来自实际场景，不能根据环境 ID 或创建顺序推导。内部 padding 不应泄漏为额外公共控制。Step、reset 和查询使用绑定数组与句柄，不解析源资产。

`model_file` 与非空 `entity_assets` 互斥。完整模型的 `fixed_variant_plan` 不能与 `entity_assets` 同时使用；`entity_variant` 必须依附于后者。既有 `model_file`/完整模型 variant 调用保留原语义。**一份模型文件不一定是一个 articulation：** 它可能包含多个独立 root。后续归一化层必须检查实际编译分区并保留既有支持行为，不能假设每文件一个 root。单实体和多实体执行应收敛为同一映射运行时；本决策不引入第二套永久兼容 runtime。

由于 `SceneCfg` 仍可变，声明门控在消费时再次校验。在当前切片中，所有具体 adapter 经 factory 或直接构造路径均拒绝新的实体物化声明。公共词汇不等于 runtime capability：发布这些类型不能把 `entity.multiple` 或相关 variant/reset 操作标记为已实现或已通过真实验证。

## 后端实施与证据边界

| 实施 owner | 必须交付的物化与证据 |
| --- | --- |
| 共享布局/IPC（B） | 版本化 entity/state/action layout、握手和局部写入；区分 nu 与全部 DoF。 |
| IsaacGym（C） | 多原生 actor、真实全局 actor/body/DoF 索引、实体级资产选择和 indexed reset，附独立身份检查。 |
| IsaacSim（D） | 独立 articulation/rigid view、静态/视觉 prim；复用经过审查的 #79 转换/状态切片，补齐被动 articulation 和绑定 runtime 的实际实例身份验证。 |
| MuJoCo/MJWarp（E） | 冷路径组合和编译地址映射；各 variant 默认值、mass/inertia/COM 和几何派生字段；独立源/native reference。 |
| 其余五 adapter（E） | 逐组合评估、unsupported/unknown 诊断和后续 owner；拒绝不等于功能交付。 |
| 一致性验收/下游（F） | 最终 head 的真实运行记录、完整场景 playback，以及通过公共接口运行的小型共享操作场景。 |

Worker echo 验证双方对请求的理解一致，不能独立证明物理实例使用了哪份资产。证据必须连接声明身份、实际原生实例/view 以及其参数或物理响应。写入和读回复用同一错误映射不构成有效的独立 oracle。

后续必要测试包括非连续/乱序环境 ID、实体/variant 创建顺序重排、被动关节不增加 action、非单位 root 姿态和 COM 偏置、不 step 时的 reset 保留、镜像/环境隔离，以及显式非 round-robin assignment 或诚实拒绝。编译参数与独立源比较，短程批量运动与同后端独立场景比较。不要求复杂接触轨迹跨引擎数值一致。

证据按 M1 规则保留精确最终 SHA、dirty 状态、命令、引擎/adapter 版本、硬件、容差、原生读回来源和未验证范围。既有 M1 runtime 通过只证明原有 profile。契约测试、mock、skip 和 SDK 可用性不能验证新的组合 runtime。本 ADR 不完成任何 B–F 能力。

## 替代方案与影响

| 替代方案 | 决策与原因 |
| --- | --- |
| 每种工具 variant 重复完整场景 | 保留实体级声明；后端内部 realization 仍可复用既有 executor，不把重复 robot/table 资产的负担交给调用方。 |
| 全局 plan 加实体 consumer 标志 | 使用一个显式 binding，避免目标与 plan 在并行声明字段中不一致。 |
| 立即接受任意异构拓扑 | 从可校验的公共布局保证开始；可变动作/状态 shape 和原生限制需要独立测试后支持。 |
| 将 #79 整体复制为公共契约 | 复用范围明确的 adapter 机制，避免把 PhysX/importer 细节和任务 parity 设置变成通用实体语义。 |
| 将 reset 视为若干独立写入 | 提交前校验完整请求，不可恢复的原生部分失败进入 faulted，让保留规则和失败行为可审查。 |
| 能拒绝非法输入就声明所有后端支持 | 分开声明接受、实际实现与 runtime 证据。 |

这引入了公共声明面和每个实现 adapter 的责任。当 B–F 逐步交付物化、映射、执行和证据时，不支持的组合继续保持不可用。它不引入通用资产 IR、动态拓扑替换、引擎依赖、DR provider 生命周期或常规九引擎 GPU CI。
