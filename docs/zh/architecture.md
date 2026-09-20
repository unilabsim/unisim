# 架构

MJWarp 的 body getter 与 generalized-state getter 在 `step()` 返回后处于同一边界，callback 与无 callback 路径一致。首个 tracked-body getter 基于最终 qpos/qvel，在活跃的逐世界模型上刷新 tracked body 位姿与速度，覆盖 reset randomization 和 fixed-variant 行；仅读取控制状态、以及 body 状态从未被读取的步不支付该刷新开销。四个注入的 sensor 块通过构造时分配的缓冲合并为一次设备到主机传输。该刷新不会重新运行约束求解：具名 contact 与 force sensor 继续表示刚完成物理子步的值。

[English](../en/architecture.md) | [中文](architecture.md)

MuJoCo 与 MJWarp 适配器的 `get_state()` 快照使用模型完整的 MuJoCo generalized-state 布局（`nq` 列 qpos、`nv` 列 qvel）。它与 `set_state()` 接受的布局以及具名状态和浮动根索引 API 的列号一致。因此固定基座模型不会合成 root 列，模型中其它位置的 free joint 也保留原生 qpos/qvel 位置。

`get_joint_range()` 不传参数时返回既有关节限位表。传入 `names` 时，MuJoCo 与 MJWarp 适配器按名称选择 hinge 或 slide 关节，保留请求顺序，并返回 `(N, 2)` 位置上下限；hinge 使用弧度，slide 使用米。未知名称和非标量关节以 `ValueError` 快速失败；未启用限位的标量关节返回 `[-inf, inf]`。名称到模型的映射保留在适配器内部，并与 qpos 地址无关。

`unisim-core` 拥有公共物理契约、后端能力、适配器工厂边界、引擎原生资源、一致性检查，以及预留的 benchmark case/result schema。它不依赖 UniLab、Hydra、Torch、Gymnasium、learner、runner 或任务代码。

UniLab 拥有 task/env/manager 生命周期、Hydra owner YAML、机器人资产、训练、checkpoint 和 sim2sim 策略 I/O。UniLab 将任务拥有的场景与随机化输入转换为 UniSim 契约。

引擎适配器使用包中立的 `SceneCfg` 和可选的向量化环境数量构造。模型元数据只在构造期间解析；状态和控制数组通过公共契约传递，引擎对象不会逃出适配器。IsaacGym 与 IsaacSim 共享子进程 IPC 组帧，并把它们的 Python 3.8 与 Kit worker 保留在核心 wheel 之外。

`unisim.ADAPTER_SPECS` 是所有已声明 UniLab 后端身份的唯一迁移清单。`available` 表示存在公共适配器和诊断；它不表示每台主机都安装了专有 SDK 或 GPU 运行时。运行时支持由适配器的 optional-extra 与 worker 冒烟测试确立。

所有资产和模型元数据解析都是冷路径职责。热路径 `step` 与 `reset` 代码接收校验过的数组和缓存标识符；适配器不得动态探测引擎私有属性。

区间域随机化是声明式的：UniSim 的 manager 从 `unisim.dr.interval` 定义的 `IntervalTermOp` 描述符构建 `IntervalRandomizationPlan.ops`（项名称、NumPy 载荷和可选 body ID；只使用标准库与 NumPy，因此计划在基于 spawn 的 collector 进程间保持可 pickle）。每个后端拥有自己的能力声明（`supported_interval_terms`）和冷路径构建的 `_interval_term_handlers()` 表；通用 `SimBackend.apply_interval_randomization` 分发会用内建项规格校验每个 op，路由到匹配 handler，并对后端未声明的项以 `NotImplementedError` 快速失败。自定义项是由注册后端拥有的自由字符串，只根据该后端的能力集校验。

重置时的模型字段写入使用 `ResetRandomizationPayload` 上的精选字段；调用者绝不独立提交几何包围盒等由编译器派生的字段。适配器通过 `SimBackend.get_reset_term_default(term)` 暴露权威默认值：单模型后端返回规范表，固定变体建立不同基线时返回逐环境表。默认值查询绝不随 reset 随机化改变。

body 质心偏移（`body_ipos`，即质心在各 body 局部坐标系中的位置，单位为米）有两种明确的查询形式。不带参数的 `SimBackend.get_body_ipos()` 在任何模式下都返回形状为 `(nbody, 3)` 的规范模型默认表；`get_body_ipos(env_ids=...)` 返回当前生效的逐环境值，形状为 `(len(env_ids), nbody, 3)`，顺序与 `env_ids` 一致，反映迄今为止应用的所有 reset 随机化（包括与 `base_com_offset` 的组合），局部 reset 未触及的环境保持之前的值。索引必须是一维整数序列；保留顺序、重复项和空选择，小数、布尔、多维及越界索引以 `ValueError` 失败。MuJoCo 与 MJWarp 适配器实现两种形式；mapped IsaacSim 的规范形式来自编译源，选择形式来自已校验的原生物化记录，其 legacy model-file 路径快速失败。

逐环境重力是 MuJoCo 与 MJWarp 两个适配器上的一等 reset 项。MJWarp 在与其模型字段相同的冷路径扩展中铺开逐世界的 `opt.gravity` 向量；每个消费内核都按 world 索引它，因此一次 reset 行写入会在该世界的 reset 后 forward 中生效，并持续到下一次显式更新。

外部 body wrench 在支持该能力的适配器上保持同一生命周期。`apply_body_force()` 接受可选的世界系力矩通道（力与力矩作用于目标 body 的质心，与 MuJoCo `xfrc_applied` 语义一致），`body_force` 与 `body_torque` 区间项共享同一暂存区，同一控制步内的多次提交可叠加，暂存 wrench 会作用于下一次 `step()` 调用的每个子步，随后被消费。新的 plan 会替换尚未消费的旧 plan，局部 reset 只清理被 reset 世界的暂存行。Mapped IsaacSim entity scene 使用 IsaacLab 外部 wrench 缓冲实现该固定区间生命周期，并可与下述动态 callback wrench 叠加合成。

`SimBackend.set_pre_step_control()` 在每个物理子步前转换策略控制。回调还可以额外返回携带动态 body wrench 的 `PreStepControlOutput`：wrench 每个子步从头重算（绝不跨子步或跨控制步累积），与暂存的区间 wrench 叠加合成，并在 `step()` 调用结束时一并清理。支持 wrench 的适配器会在 callback 路径内把 tracked-body 世界状态刷新到子步起始状态——MuJoCo 上使用 mjbatch 的 split-substep 传感器增量拷出（`mjbatch-uni >= 0.2.1`，硬性执行器要求），在每个子步边界以 memcpy 代价刷新 tracked 世界系传感器视图；在回调内部调用区间/力暂存 API 会快速失败，回调必须改为返回自己的 wrench。仅支持 ctrl 的适配器对 wrench 返回值抛出 `NotImplementedError`，而不是静默降级为只取 `ctrl` 分量。

上述 callback 段落描述的是支持动态 callback 的适配器。Mapped IsaacSim entity scene 通过每个公开物理子步执行一次 worker STEP，并在每次 callback 前刷新共享状态来支持宿主 callback；这是正确性路径，不是 #152 跟踪的 device-resident controller 或性能路径。legacy IsaacSim model-file callback 仍不支持。

MuJoCo factory 选项 `refresh_pre_step_body_state` 只控制上述 callback 时间的 tracked-body 刷新。默认值 `True` 保持上述契约；显式传入 `False` 时仍保留注入的 body sensors 和 body-state getters，广义状态与动态 wrench 仍逐子步更新，但 callback 内的 body-sensor 视图不保证新鲜。这样依赖广义状态的控制器可以在 `implicitfast` 积分器下运行，无需访问执行器私有选项。

在 MJWarp 上，callback 循环仍在 host 执行，但每次上传后的物理子步在可用时回放已捕获的固定地址 step graph；捕获不可用或失败时保留 eager kernel launch。Graph 回放只是执行调度优化，不承诺与 eager 输出逐位一致：短时程 float32 比较必须说明容差，而重接触轨迹可能放大微小的执行顺序差异。

MuJoCo 控制步返回后，tracked-body getter 会在首次读取时与该步最终的 `qpos`/`qvel` 同步。getter 仅对请求中仍需同步的环境行执行主机侧 kinematics 与 velocity-sensor 重算，复用 materialize 时分配的一个 scratch `MjData`，并合并拷贝相邻的注入 sensor 块。局部 reset 和速度更新只清除受影响行的待刷新标记；callback getter 使用执行器的拷出结果，不重复执行主机侧运动学。未请求 body tracking 的环境、body 状态从未被读取的步，以及只读取广义状态的路径不支付刷新开销。用户声明的 sensors 保持原生 `mj_step` 之后的时序：contact force 等 acceleration-stage 值保留最后一个物理子步实际求解出的结果，不会被最终状态的完整 `mj_forward` 替换。

如果 pre-step callback 抛出异常，MuJoCo 与 MJWarp 会消费暂存和动态 wrench，同时保留已完成的子步。公共状态仍可读取以便恢复执行；MJWarp 还会发布已完成子步对应的经过时间和 sensor 缓存。清理不会回滚物理状态，也不会重新求解受力。

固定模型身份与重置随机化分离。完整模型任务在 `SceneCfg.fixed_variant_plan` 上携带计划；portable entity 场景通过 `SceneCfg.entity_variant` 携带同一计划，其 catalog sources 描述被绑定的目标实体而不是完整场景。引擎适配器在构造期间、首次 forward 之前以及 CUDA graph 捕获之前实现选中的身份。该计划包含最终只读赋值行、完整物化的 `ModelSourceDescriptor` 条目，以及公共布局保证（`same_layout` 或 `uniform_public_layout`）。域随机化能力声明适配器能实现的布局，以及播放是否暴露逐环境模型。计划与能力对象只使用标准库和 NumPy 类型，因此保持可 pickle；活跃 `MjSpec`、mjbatch 与 Warp 对象绝不跨越该边界。槽位合并、mesh/material 池化、逐世界数组、派生字段重算和播放表示都是适配器拥有的实现细节。

MuJoCo 适配器实现该契约，同时没有重新引入每个环境一个完整模型。在冷路径上，它独立编译每个物化 MJCF 作为数值/默认值 oracle，校验 same 或 uniform public layout，并把规范 mesh 池化委托给 mjbatch 的 `VariantPack`。对于 entity binding，可选槽位校验覆盖目标实体及其显式镜像；无关场景 geom 必须保持稳定，但不定义目标槽位。`uniform_public_layout` 只允许命名 mesh-geom 槽位缺失，选择 geom 数最多的实现作为规范源（并列时取较低索引），只在该规范源上池化 catalog mesh，并在构造 pack 前规范化派生 body 语义。适配器保留规范执行器模型、由编译器派生的变体行、紧凑默认值表和不可变赋值，而不是每个变体一个已编译 `MjModel`。运行时重置写入使用与规范模式相同的扩展模型字段视图。Same-layout mesh-geom 槽位可以逐世界禁用，而公共状态或控制拓扑变化会快速失败。播放按需从所选源编译分离的视觉 oracle，离线播放为每个被渲染环境保存一个自包含模型。

MJWarp 适配器在构造期间实现该计划。它独立编译每个 MJCF 源以获得 oracle 值，校验声明布局，把 mesh 与 material 池化到一个规范模型，并在 `put_model` 之后、首次 forward 与 CUDA graph 捕获之前安装逐世界 `geom_dataid`、`geom_matid` 和依赖 mesh 的模型字段。因此，重置镜像和 `get_reset_term_default()` 从每个世界分配到的变体开始。

IsaacGym 适配器通过 actor 级资产选择实现同一计划。whole-model plan 由 worker 用 `gym.load_asset` 把每个完整 MJCF 源各装载一次；entity binding 则先由公共 compiler 校验 portable layout，并为目标实体及其镜像导出按 assignment 选择的 expanded sources，无关实体保持 canonical。worker 校验公共 body/dof 拓扑，按不可变 assignment 行创建每个环境的 actor，并审计原生 asset identity。逐变体执行器属性与任务初始 keyframe 按关节名映射，handshake 回显 assignment，原生渲染本身展示的就是该环境分配到的 actor。内部 PhysX shape 数量可以不同，包括可选 mesh 槽位存在或缺失，但公共 state/action/sensor/dof/body 布局漂移会快速失败。该适配器仍未声明 reset-time model-field 随机化，因此该能力不可用。

语义边界见[能力与证据 ADR](adr-capabilities.md)，portable 场景 authoring 见[可移植 MJCF ADR](adr-portable-mjcf.md)。无需 SDK 的声明、限定范围的证据、缓存导入报告和显式严格构造独立于 `AdapterSpec.status`；[自动生成语义清单](support-matrix.md#语义清单) 描述源码审查子集及未明确支持的特性。
