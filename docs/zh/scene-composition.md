# 声明式场景组装与固定变体池

[English](../en/scene-composition.md) | [中文](scene-composition.md)

状态：已接受的契约，用于 IsaacSim 子进程适配器上的多资产 URDF 场景。本文档既是该契约扩张的决策记录，也是其字段、fail-closed 行为与握手协议的参考。

## 决策记录

以下六个问题决定公共面的扩张边界，决策如下：

1. **固定变体计划可以绑定到单一场景实体吗？** 可以，通过 `SceneEntitySpec.consumes_fixed_variant_pool`。每个场景恰有一个实体可声明该绑定，host 冷路径对称校验：计划没有恰一个消费实体、消费实体（或镜像声明实体）没有计划、或消费实体的形态无法承载刚体对象池，都在任何 worker 启动之前失败。模型级实现（IsaacGym/MJCF 通道）与实体绑定实现（IsaacSim/URDF 池）共享 `SceneCfg.fixed_variant_plan` 这唯一的构造期输入。
2. **`entity_assets` 是 backend 中立还是 IsaacSim 私有？** 意图上 backend 中立，能力上单一消费者。类型化声明放在 `SceneCfg` 上，因为场景组装是 owner 配置而非适配器内部；当前只有 IsaacSim worker 物化它，其余后端在构造期拒绝声明了该字段的场景而不是忽略它。第二个消费者将复用同一声明而非引入新声明。
3. **不支持的后端如何 fail closed？** 在构造期，通过 `validate_scene_composition_support`。场景组装字段（`entity_assets`、`ground_plane`、`physx`、`env_grid_spacing`）是声明而不是提示：无法物化它们的后端在构造时抛出 `NotImplementedError`。这保持公共面不变（不新增 capability 对象），同时消除所有静默降级路径。子进程家族通过 `_supports_entity_assets`/`_supports_ground_plane`/`_supports_scene_physx`/`_supports_env_grid_spacing` 钩子传递这些旗标，IsaacSim 特化自行声明。
4. **`FixedVariantMetadata.mass` 是通用公共需求吗？** 是，作为被测量的量。URDF 资产把质量写在 inertial 块里，但 USD 转换会烘焙它，因此真正仿真的质量只有后端权威可知。`SimBackend.get_entity_variant_metadata(entity)` 返回 worker 实测的按环境质量表；assignment 与 scale 不进入公共类型（它们是 owner 已持有的构造期输入）。
5. **刚体实体根状态与 wrench 的公共语义是什么？** 每个声明的刚体实体拥有一条 13 宽的根状态行（世界系 xyz、wxyz 四元数、世界系线速度与角速度），通过每实体的共享内存槽位与 `set_state(entity_root_states=...)` 暴露。一条 `set_state` 命令是对同一批环境行的一个事务；仅刚体根的 reset 不会扰动 articulation。世界系的力/力矩 wrench 经由 `apply_body_force` 在同样的根上暂存：在一个 interval 计划内累积，作用于下一个控制步的每个子步，随后被消费；reset 会取消所选环境已暂存的行。
6. **地面平面的消费能力如何声明？** 以存在性声明：`SceneCfg.ground_plane` 声明一个世界级地面（摩擦/恢复系数/范围），IsaacSim worker 将其物化为离线安全的碰撞地面。未声明的场景保持后端的原生地面行为。场景模型自带地板的后端（MuJoCo 家族）在构造期拒绝该声明——那里的地面通道就是 MJCF 文件本身，声明会被静默丢弃内容。

## 场景声明

| 字段 | 类型 | 默认 | 消费者 |
| --- | --- | --- | --- |
| `entity_assets` | `tuple[SceneEntitySpec, ...]` | `()` | IsaacSim |
| `ground_plane` | `GroundPlaneSceneCfg \| None` | `None` | IsaacSim |
| `physx` | `ScenePhysxCfg \| None` | `None` | IsaacSim |
| `env_grid_spacing` | `float \| None` | `None` | IsaacSim |
| `fixed_variant_plan` | `FixedVariantPlan \| None` | `None` | IsaacSim 实体池、IsaacGym 模型级 |

`ScenePhysxCfg` 逐字段镜像 IsaacLab 的 `PhysxCfg`（求解器类型、迭代钳制、反弹阈值、摩擦偏移/相关距离、GPU 接触流缓冲）；声明它即显式选择场景级求解调优，未声明的场景保持后端自身默认。`env_grid_spacing` 以米为单位声明环境克隆网格间距（未声明保持原生 2.0 m 布局）。两者都不是提示：不能消费它们的后端在构造期拒绝该场景。

## 实体声明

`SceneEntitySpec` 声明一个逻辑资产角色：源文件、格式标签（`urdf`/`mjcf`）、物化方式（`articulation`/`rigid`）、根模式（`fixed`/`floating`/`kinematic`），以及可选的组合字段：

- `actuator_gain_overrides`：按关节的 PD/动力学表（URDF 资产不带增益）；未知关节名在扫描期失败。
- `contact_friction` 与 `contact_friction_by_body`：PhysX 接触材料默认值与按 body 覆盖，经运行时 PhysX 视图写入并在 INIT 时 fail-closed 读回校验。
- `init_state`（`EntityInitStateCfg`）：articulation 出生位姿。固定基座机器人的根位姿没有其他写入通道，因此这就是出生通道；未声明保持后端默认出生位姿。仅限 articulation 实体。
- `collision_enabled`：USD 烘焙碰撞旗标。`None` 保持转换后 USD 的碰撞状态；`False` 为非物理视觉替身关闭碰撞。
- `replace_cylinders_with_capsules`：URDF 转换器旗标。`None` 保持按物化方式的默认（floating rigid 实体以胶囊替换转换——动态对象契约；其余保持转换器默认）。
- `consumes_fixed_variant_pool`：把该实体的资产绑定到场景的固定变体计划（每场景恰一个实体）。
- `mirrors_fixed_variant_pool`：镜像池目标按环境变体的 kinematic rigid 视觉替身。与消费池互斥，仅对 rigid kinematic 实体合法。

不从实体名称推断任何任务语义：worker 依据声明的物化方式/根模式与声明的碰撞旗标导出烘焙计划，绝不依据角色名。

## 变体池通道及其握手

`fixed_variant_plan` 在 IsaacSim 上的实现是在任何 worker 进程启动之前完成暂存的实体绑定刚体对象池。host 侧暂存（`build_init_variant_pool_payload`）fail-closed 校验：恰一个消费绑定、rigid floating 的 URDF 目标、`SAME_LAYOUT` 计划布局、仅 round-robin 指派，以及——通过 host 侧 URDF 扫描——可解析、单根、无可动关节、且整个目录与目标 bootstrap 资产的根/体布局完全一致的源文件。校验后的池随 INIT 载荷下发；worker 对每个源转换一次、烘焙物理、从烘焙后的 USD 测量每个变体的质量，并以 `K` 个唯一原型（`MultiUsdFileCfg`、`random_choice=False`）物化池：环境 `i` 按构造取得源 `i % K`。

INIT 握手是权威性的：worker 回传 `fixed_variant_count`、`fixed_variant_assignment` 与 `fixed_variant_target_entity`，host 逐一比对数量、目标与完整指派和不可变计划。舞台取证（`variant_assignment.observed`）保持为可选诊断——当 prim 栈无法遍历时 `None` 可接受，但已计算出的观测与回传不一致时 fail closed。身份是构造期的：reset 永不重指派变体。

能力协商对池化场景声明 `supports_fixed_variants`（`SAME_LAYOUT` 布局）与 `supports_per_env_playback`；`get_playback_model(env_index)` 将每个环境解析到其被指派的变体源，并要求显式环境下标。`get_entity_variant_metadata(entity)` 按环境展开 worker 实测的质量表。

## 证据与边界

host 侧覆盖位于 `tests/adapters/isaacsim/`（池暂存、含篡改场景的握手守卫、源预校验、布局与 round-robin 门、playback 解析、质量读回、组合 fail-closed）与 `tests/contract/test_scene_composition_fail_closed.py`（跨后端构造门）。Kit 层行为（USD 烘焙读回、场景 PhysX 生效、池物化、wrench 语义）由工作区探针套件而非 pytest 验证。

已记录为后续事项而非静默缺口的边界：内容寻址的 URDF→USD 转换缓存（每次 INIT 都重转换）、池镜像的 convert-once copy-per-role（镜像批会二次转换源）、以及任意（非 round-robin）指派的精确 K 原型 spawner——它今天以可操作的报错 fail closed。仅支持 round-robin 指派与单刚体 URDF 变体；articulation、固定根或 kinematic 池目标与 `UNIFORM_PUBLIC_LAYOUT` 计划被拒绝。
