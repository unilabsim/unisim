# MuJoCo 实体组合与局部重置

[English](../en/m2-mujoco.md) | [中文](m2-mujoco.md)

## 范围与 owner

本文描述 [#108](https://github.com/unilabsim/unisim/issues/108) 的 MuJoCo CPU 实现切片，遵循 [M2 实体决策](adr-m2-entities.md)。实现组合 MJCF 实体源，通过既有 `mjbatch.Batch`/`VariantPack` 运行完整场景，不为各实体引入独立仿真器，也不证明其他引擎已支持。

`backend/mujoco/composition.py` 负责冷路径源归一化、命名空间 attach、编译布局和临时完整场景产物。`backend/mujoco/backend.py` 负责 batch 状态、原生映射、reset 和 playback。UniLab 继续负责注册资产、任务 selector、策略 I/O 和 checkpoint 兼容。构造完成前源目录须保持可用；backend 持有生成产物直至 `close()`/场景清理。

## 支持的声明切片

| 声明 | MuJoCo CPU 行为 |
| --- | --- |
| 物理 MJCF 实体 | 一个具名 root body，非 root 关节具名，无 world 层 geom；多个实体 attach 在各自 `<entity>/` 下。 |
| 浮动 articulation 或 rigid object | 源根含一个 free joint；rigid 源不能包含非 root 关节。 |
| 固定 articulation 或静态 rigid object | 源根没有 root joint 或 mocap 标志；被动子关节保留状态但不增加动作。 |
| Kinematic rigid object | 使用 mocap root，不保留 free-joint 动力学或 actuator。 |
| 视觉镜像 | 继承目标源和 fixed identity，位姿独立，使用 mocap，移除关节、actuator、sensor 和碰撞参与。 |
| 实体 variants | 一个物理 consumer，要求 `same_layout`，保留任意合法 assignment；每个 catalog 项生成独立编译的完整场景。 |
| 具名源 keyframe | 合并所有物理源的 key 名称；每个 realization 保留其关节、控制和 activation 值。 |

各源的全局物理选项必须相同；实体间和 variant 间的冲突均拒绝，不依赖声明顺序。Factory 的 `sim_dt` 显式覆写源 timestep。写出生成 XML 前解析相对 mesh/texture 引用。声明的基础源与所有 catalog 项须具有一致的编译公共布局，同时检查 sensor 布局、key 名称和 activation 宽度。

当前 adapter 对新实体入口拒绝非 MJCF 源、`uniform_public_layout`、源 tendon/equality、未命名 keyframe、fragment、terrain 组合及 `visual_model_file`。也拒绝 attach 无法可靠保留的非默认 compiler 设置，包括 `settotalmass`、质量/惯量边界、静态融合和视觉丢弃。既有完整模型入口保留原有支持。拒绝不意味着 MuJoCo 引擎本身缺少这些能力；这些组合需要显式 adapter 实现和证据。

## 初态与 keyframe

Root pose 始终来自 `EntityInitialState`，使用环境世界系的 link 原点 xyz 和单位 wxyz。初始 root 速度为零。这些规则覆写原始源根位姿及每个源 keyframe 的 root pose/velocity。镜像保留自身声明的 mocap 位姿，不跟随目标位姿。

场景 key 集合是所有物理源具名 key 的并集。对每个 key，adapter 使用实际关节宽度和 activation 地址复制编译后的非 root 关节 qpos/qvel、actuator ctrl 和 activation，不依赖猜测的标量关节数组拼接。缺少某 key 的实体提供自身源 qpos0 和零 qvel/ctrl/act。同名 key 的时间必须有限且相等。设置 `default_keyframe_name` 时，该 key 必须存在于每个场景 realization；未设置时仍使用源 qpos0 默认路径，不静默选第一个 key。

绑定 batch 前，backend 为每个 variant/环境分别准备所选默认值。完整 `reset(env_ids)` 恢复各环境默认值，包括控制、activation 和 mocap 位姿。`reset_entities()` 只应用显式 patch 字段；缺失字段保留当前值，两种 reset 都不改变 variant identity。

## 公共状态与 reset

`get_scene_layout()` 返回冻结的编译公共地址；`get_entity_names()` 和 `get_entity_state(name)` 提供具名 root/joint 状态。返回数组与内部存储分离。浮动 root 的 pose/velocity 来自当前广义状态，并将 MuJoCo 的 body-frame 角速度转换到世界系。固定 root 使用编译位姿，kinematic root 使用 batch 的 mocap 状态。Root 线速度参考 link 原点，不是 COM。

`reset_entities(SceneResetRequest(...))` 在第一次写入前校验完整请求并准备临时行，只 scatter 请求的列与 mocap 项，清理与所改状态关联的控制/activation、施力和 warmstart 通道，只对选中环境执行 forward。其他实体的通道和未选环境保持不变。Scatter 保留请求行顺序；原生 `forward(ids)` 按 mjbatch 要求接收排序后的 ID。只写 pose 时，也将保留的世界系角速度转换到新 body frame。

局部实体 reset 不调用整环境 `Batch.reset()`。原生提交失败可能留下部分写入状态，因此 backend 进入 faulted 并拒绝继续 step 或消费状态，必须重建。提交前的校验错误保证状态不变。

既有完整状态 `set_state(env_ids, qpos, qvel, ...)` 保持整环境 reset 语义。它与 `reset_entities()` 现为同一个 adapter-owned `StateCommitPlan` 提交器准备不同 intent。完整 reset 清理选中 world 的时间、控制、activation、pending/applied force 和 warmstart；实体 patch 保留无关通道。模型写入和状态 shape/finite/range 检查在首次原生写前完成。负质量/惯量、armature、damping/frictionloss、几何尺寸/摩擦和非单位惯性四元数在准备期拒绝；保留 solref/solimp 的特殊符号语义。需要保留其它实体时仍使用 `reset_entities()`；共用执行并不使整环境与局部 reset 语义相同。

## Playback 与生命周期

`get_playback_model(env_index)` 返回所选环境独立编译的完整场景，包含机器人、物体、静态几何和镜像。组合场景含 mocap body 时，`get_physics_state()` 在尾部附加 `[mocap_pos, mocap_quat]`。既有渲染器消费该尾部，保留被独立移动的视觉目标。没有该尾部的场景维持原 snapshot 表示。

Backend 在生命周期内持有生成 XML，关闭或构造失败时清理。源解析、key 合并及身份/布局校验属于构造工作；step、局部 reset 和报告读取不重新解析资产。

## 构造报告与 provenance

既有 [M1 报告 schema](adr-m1-capabilities.md) 增加 adapter 自有字段，不引入新 schema 或能力注册表。每条记录按实体、catalog variant 及被分配到该 variant 的环境限定范围；未使用的 variant 对应空环境集合。

| 字段 | Requested/effective 值及来源 |
| --- | --- |
| `entity.source_root_pose` | 原始独立编译源的 root pose → 组合编译默认位姿。来源区分 source 与编译模型读回。 |
| `entity.keyframe_roots` | 原始具名 key 的 root pose/世界系速度 → 经声明位姿/零速度覆写后的组合 key 值。这是编译 key 值，不是当前仿真状态。 |
| `entity.initial_defaults` | 所选源 key 或源默认值 → adapter 准备的初态/reset root/joint/ctrl/act 值。Effective 来源为 `adapter_setting`，不是原生 batch 读回。 |

不同值标记 `overridden`，一致值标记 `exact`。镜像记录显示继承源的 joint/control/activation 被移除及其独立位姿。报告是不可变的构造缓存快照，不随 reset/step 改变，也不能证明后续实时状态等于默认值。原生 batch 验收是独立证据；这些字段不会自动建立 runtime-verified capability。

## 验证与剩余验收

从 UniSim 仓库根目录运行 focused 命令：

```bash
uv sync --locked --extra mujoco
uv run --no-sync pytest -q tests/adapters/mujoco/test_composition.py tests/adapters/mujoco/test_entity_runtime.py
uv run --no-sync pytest -q tests/adapters/mujoco
```

Fixture 使用真实 MuJoCo/mjbatch CPU：N=2/K=2 与 N=5/K=2、assignment `[1,1,0,1,0]`；固定/浮动机器人和被动物体；原生质量/惯量及带 COM 偏置的独立 link 速度读回；局部 reset 保留控制、activation、力、warmstart 和未选行；pose-only 保留角速度；具名默认值；完整 playback；有/无镜像 rollout 对照；独立原生单环境 rollout；物理接触和环境隔离；校验/原生失败行为。构场测试覆盖相对 mesh、全局选项冲突、key 宽度/时间及资源清理。

合入门禁必须在最终 head 运行仓库检查，保留精确 SHA、引擎/runtime 版本、硬件、命令、容差和未验证范围。工作树测试可作为实施证据，不能当作最终 head 验收或跨引擎等价证明。MJWarp 和 Isaac adapter 仍须在 #108 下交付各自实现和真实运行结果；本 CPU 切片不关闭整个里程碑。
