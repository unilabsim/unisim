# 适配器支持矩阵

[English](../en/support-matrix.md) | [中文](support-matrix.md)

| 后端 | 公共类 | 安装与运行时边界 | 状态 |
| --- | --- | --- | --- |
| MuJoCo | `unisim.MuJoCoBackend` | `uv sync --extra mujoco`（mjbatch 运行时） | available |
| Motrix | `unisim.MotrixBackend` | `uv sync --extra motrix` | available |
| Drake | `unisim.DrakeBackend` | `uv sync --extra drake`（`drake-uni`）及其原生批处理扩展 | available |
| MJWarp | `unisim.MJWarpBackend` | `uv sync --extra mjwarp`，CUDA | available |
| Genesis | `unisim.GenesisBackend` | `uv sync --extra genesis`（`genesis-world==1.3.3`） | available（原生 CPU 证据） |
| Newton | `unisim.NewtonBackend` | `uv sync --extra newton`，Newton 1.5.1 与 MuJoCo-Warp 3.11.0 | available（CUDA） |
| SuperDex | `unisim.SuperDexBackend` | `uv sync --extra superdex`，CPython 3.12 或 3.13，SuperDex 1.3.0 | 实验性 CPU；见[配置说明](superdex.md) |
| IsaacGym | `unisim.IsaacGymBackend` | `uv sync --extra isaacgym`（空 extra）加专用 Python 3.8 worker | available |
| IsaacSim | `unisim.IsaacSimBackend` | `uv sync --extra isaacsim`（空 extra）加专用 IsaacSim 或 IsaacLab worker | available |

基础 wheel 不导入以上任何 SDK。构造执行冷路径运行时发现，并在运行不可用时抛出适配器专属、可操作的错误。本矩阵是适配器与 API 支持声明，不是每台主机都具备每个厂商 SDK 或 GPU 能力的声明。

SDK-free 的 portable MJCF compiler contract 可随基础包导入；实际冷路径编译按需要求 `unisim-core[scene-compiler]`（`mujoco~=3.11.0`，不包含 mjbatch executor）。Compiler 的 source/intent report 与内容身份本身不声明 native adapter 支持；每个 adapter 仍需自己的物化与读回证据。治理边界见[可移植 MJCF ADR](adr-portable-mjcf.md)。

IsaacSim 的 raw/role 派生 USD 缓存只是冷路径物化优化。它们不缓存原生场景、view、参数或 effective report；命中仍基于缓存 USD 物化并读回。缓存根目录、环境覆盖、身份输入、role 校验与原子发布行为见[实体场景执行](entity-scenes.md)。

mapped IsaacSim 场景暴露冻结的公开 geometry 名称、归属 body ID、规范化原生 collider mask，以及逐环境当前 PhysX 摩擦 material。mapped `set_state()` 的局部行支持逐环境 reset 随机化：正 `body_mass`、`body_ipos` 与正对角 `body_inertia`、Coulomb `geom_friction`、actuator `kp`/`kd` 驱动增益，以及非负的 `dof_damping`/`dof_armature`/`dof_frictionloss` 关节值（free-root DOF 列保持为零）；`base_mass_delta`/`base_com_offset` 在 host 侧折叠进主实体根 body。每次写入都从原生 PhysX view 读回并在 reset barrier 返回前与请求逐项核对。`FixedVariantPlan` 各 variant 的驱动增益可以不同，并在 spawn 时按环境写入。gravity、惯量主轴方向、geometry 尺寸与 solver contact 参数带显式理由 fail closed。legacy model-file 场景快速失败。源意图/materialization/当前值边界见[实体场景执行](entity-scenes.md)。

IsaacSim 在构造时接受一组有边界的 PhysX solver 配置：factory 选项 `isaacsim_solver_position_iteration_count`、`isaacsim_solver_velocity_iteration_count`、`isaacsim_bounce_threshold_velocity` 与 `isaacsim_contact_offset`（或 `IsaacSimBackend` 上同名但不带前缀的关键字参数）。position 迭代次数必须是正整数，velocity 迭代次数必须是非负整数，bounce threshold 必须是非负有限浮点数，contact offset 必须是正的有限浮点数；非法值在任何 worker 启动前快速失败，worker 也会重新校验 INIT payload。迭代次数会同时固定 PhysX 场景的 min/max 区间，使每个 actor 都被钳制到请求的次数；bounce threshold 映射到 IsaacLab 的 `PhysxCfg.bounce_threshold_velocity`；contact offset 在冷路径 spawn 完成后写入每一个 collision shape。已配置的值会作为从 USD stage 读回的 engine readback 进入 worker 的版本化 configuration report，host 会像守护 render-mode 契约一样严格比对请求值与读回值；未配置的值保持 IsaacLab/PhysX 默认行为不变。这些运行时设置不会进入 raw/role USD 缓存 identity，因为它们不改变被缓存的 artifact。MuJoCo/SimToolReal 的 substeps 在这里不是独立的 solver 参数：subprocess 契约用 `step(ctrl, nsteps)` 的 decimation 来表达，因此 SimToolReal 默认值映射为 `sim_dt=1/60` 加 `nsteps=2`、`isaacsim_solver_position_iteration_count=8`、`isaacsim_solver_velocity_iteration_count=0` 与 `isaacsim_contact_offset=0.002`。

mapped IsaacGym 场景支持局部环境 reset randomization，覆盖 actuator `kp`/`kd`、`body_mass`、`body_ipos`、`body_inertia`、`dof_armature`、`dof_frictionloss` 与 Coulomb `geom_friction`。公共列先在宿主预校验，在写入暂存状态之前通过 PhysX actor DoF、刚体与形状属性 setter 落盘，再做原生读回并审计选中行正确性与未选中行隔离；原生写入或读回不一致会使 worker fault，只有通过校验的记录才刷新状态换算依赖的 COM 缓存。惯性写入是与所分配 variant 惯性姿态复合的主惯量三元组，armature/friction-loss 写入必须保持浮动根列为零。`gravity`（PhysX 中为仿真全局量）、`dof_damping`（PhysX 没有独立于导入 position drive 的被动关节阻尼通道）、base mass/COM delta、惯性姿态、geometry 尺寸与 solver 参数均 fail closed；legacy model-file 路径上的所有 randomization 项同样 fail closed。mapped 场景还在宿主暂存世界系 `body_force`/`body_torque` 区间 wrench，并在每个请求的子步前通过原生刚体 force tensor 重新施加，因为 PhysX 每次 simulate 都会消费外力；力作用于 body COM。固定根 MJCF 实体（`root.fixed`）在 mapped profile 下声明为 exact。确切边界见[实体场景执行](entity-scenes.md)。

MuJoCo 适配器的原生执行器是 [mjbatch](https://github.com/unilabsim/mjbatch_uni)，即 `kevinzakka/mjbatch` 的维护 fork；它为 Linux x86_64、aarch64 和 macOS（CPython 3.10 到 3.14t）提供预构建 wheel，并精确固定 `mujoco==3.11.0`。原生执行器不支持 Windows 和 musllinux，因此 Windows CI 作业只运行核心与导入边界子集。切换到当前执行器前后的数值结果不保证相同；漂移由已记录基线表征，而不是位级精确门禁。适配器支持构造时 `FixedVariantPlan` 目录及 `same_layout` 与 `uniform_public_layout` 保证。Same-layout 变体和可选命名 mesh-geom 槽位通过 `VariantPack` 合并到一个规范 mjbatch 执行器；异构公共拓扑快速失败。重置模型字段写入与逐世界编译器默认值使用 mjbatch `expand` 与 `set_const`，播放暴露逐环境独立编译的视觉 oracle。`chunk_size` 与 `adaptive_chunk_size` 是已弃用的 warn-and-ignore 参数；chunk 调度器已移除，mjbatch 的工作窃取线程池是调优机制。

MuJoCo 相关 extra 共享同一条版本线（MuJoCo 3.11、MuJoCo-Warp 3.11 和 warp-lang 1.16.0），可以联合安装。`mjwarp` 用 `mujoco-warp~=3.11.0` 跟踪该版本线，而 `newton` 保留与上游精确耦合的固定版本（`newton==1.5.1`、`mujoco-warp==3.11.0`、`mujoco==3.11.0`、`warp-lang==1.16.0`）。安装后运行 `uv run scripts/diagnostics/check_newton_runtime.py` 执行仅元数据探测；需要显式导入原生运行时时添加 `--import`。Newton 冷路径校准会采样求解器计数，并在 `nconmax` 或 `njmax` 过小时抛出显式容量错误；它绝不接受静默约束截断。

Newton 的 portable entity profile 有明确边界：它把独立的同布局 variant builder 物化到显式世界，并为每个物理实体绑定一个公开 articulation view。覆盖范围包含固定/浮动根、被动/静态实体、同 shape 类型的异构身份与力响应、局部状态 reset、带逐世界归因的具名 found contact，以及逐 variant playback。局部 reset 清空选中 control 并保留无关 control；`restore_default_controls` 与 keyframe control 恢复均快速失败。kinematic mirror 与混合 shape 类型 assignment 均快速失败。audit 与原生验证边界见[实体场景执行](entity-scenes.md)。

Genesis 的 portable entity profile 是有边界 MJCF 子集：独立固定/浮动/被动/静态物理实体与关闭碰撞的 visual mirror，通过审计后的公开名称、拓扑/visual 身份，以及适用处的物理地址绑定，且只接受与 Genesis 原生 balanced mapping 完全一致的单 link rigid 异构 assignment。mirror 使用 Genesis 公开 Kinematic entity，暴露零 qpos/DoF 与 collision mask，支持选中的公开世界 root 位姿写入，并在选中完整 reset 时恢复独立默认值；物性修改与 contact fragment 快速失败。构造通过公共逐 entity API 支持所选标量 hinge/slide default-keyframe qpos/qvel 与 actuator control；缺失 key 的 source 保留规范化标量 qpos 并保持 qvel/control 为零，原始 keyframe root 位姿/速度被忽略，且保留声明的 portable root 放置与零 root 速度。公开 geometry 暴露冷路径审计过的具名 sphere/box visual instance 名称、ID、body 归属，以及按冻结公开顺序聚合、审计后的原生 sphere/box 尺寸、完全一致的 Genesis 原生 collision mask、摩擦系数与 solver 参数；mirror geometry 没有可用于摩擦、solver 参数或 contact fragment 的原生 collision 身份。公开 DOF damping/friction-loss/armature 读回使用冷路径捕获的原生值、审计后的 qvel 地址，以及一致的 active 行/fixed-variant 值。entity 内未引用参考系的 site 位姿/运动 sensor 与场景级 world-referenced 限定 site 位姿/运动 fragment 要求所有 fixed variants 的完整 sensor 身份一致，并返回 wxyz site 四元数；site `FrameLinVel` 是世界系 site 点速度，site `FrameAngVel` 是世界系角速度，identity 姿态 accelerometer 使用干净的公开原生 IMU，并返回已完成 step 的 proper linear acceleration。场景级 fragment 的 world-referenced 限定 body `FramePos`/`FrameQuat`/`FrameLinVel`/`FrameAngVel` sensor 会把审计后的公开原生 link-origin 位姿/速度与逐 variant 源 inertial identity 组合。场景级跨 entity geom-pair found/netforce fragment 会按 assignment 路由精确原生 collision 身份，暴露已完成 step 的公共 contact 标志或作用在作者声明 geom1 上的 3 向量力，并在下一步前清空选中 reset 行。局部 state/reset 行保留无关实体与环境；选中行 body-mass/base-mass-delta、body-COM/base-com-offset、DOF damping/friction-loss/armature 与 actuator kp/kd randomization 会在公开列中预校验，并通过审计后的所属实体提交。portable 世界系 body-force/body-torque 提交通过公共 solver API 把审计后的公开 owned-body ID 映射到 Genesis solver link，并作用在每个 link COM；重复提交与同一 interval plan 内的 op 会为即将到来的原生 step 按通道独立加性累积，后续 interval plan 会替换未消耗的暂存，局部 state/entity reset 取消匹配行并保留无关 pending wrench，callback 内暂存快速失败。局部 entity reset 会清空或按 assignment 恢复选中行的默认 control，同时保留无关行与实体；不声明持久公共 control-target getter。缺失或模糊的 collision 身份、source contact sensor、同 entity pair、非一致 variant 尺寸/mask/摩擦/solver/DOF 值、物理 kinematic 实体、mirror mass/COM/DOF/actuator 修改、mirror contact fragment、旋转 accelerometer、其他 source sensor 形式、source body sensor、inertial 姿态不匹配、引用形式、其他 body/site fragment 形式、其他 contact 形式、其他 reset randomization、activation state、任意 keyframe 语义与任意力作用点均快速失败。原生 CPU 证据使用 Genesis 1.3.3、Torch 2.14.0+cpu 与 Quadrants 1.3.0；不声明 GPU 能力。精确 variant 与验证边界见[实体场景执行](entity-scenes.md)。

Motrix 的 portable entity profile 是有边界 MJCF 子集，覆盖固定/浮动/被动/静态实体、不可变 same-layout fixed variants 与有边界完整 mesh 的 `uniform_public_layout` variants。Same-layout variants 为每个被使用的 variant 拥有一个原生 Motrix model/data context，公开状态与 control 会在这些 context 之间显式 scatter/gather。完整 mesh 的 uniform-public plan 在 MotrixSim Core 0.10.1 上使用一个规范 Motrix model/data context，把受影响的公开 geom 绑定到按 fixed-variant ordinal 排列的原生 mesh variant set，审计逐行原生 mesh ordinal 读回与 render offset，并暴露逐环境原生 playback。Optional 缺失 mesh slot 与非 mesh 的物理、材质、geometry 或惯量差异均快速失败。原生布局、逐行 mass/COM、实际 geometry 尺寸和 variant control 身份均被审计；由于 Motrix 不提供惯量读回，有效惯量通过原生响应验证。世界系 body force/torque 提交通过审计后的公开 body ID 与公开原生 Link API 映射，为即将到来的 step 加性累积，且 reset 取消范围限定在受影响 body。局部 entity reset 保留无关状态与 control；局部 control 恢复与完整默认 reset 使用冷路径捕获的原生构造/default-keyframe control 以及所选 keyframe qpos/qvel。选中行 body-mass/base-mass-delta、body-ipos/base-com-offset 与标量 hinge/slide joint armature/friction-loss randomization 会在公开列中预校验，并通过所属 variant data slice 与公开 Link/Joint override 提交；free-root DOF 列保持默认值且修改被拒绝，`dof_damping` 因 Motrix 0.8.2 没有公开运行时 joint-damping override 而快速失败。entity 内与场景级 fragment 世界系 site 位姿/运动 sensor 以及具名限定 site 的世界系 Jacobian 会通过审计后的 variant context 按 assignment gather。非一致公开 control 参数或 geometry 尺寸读取、缺失原生 wrench API、actuator activation state、entity 内 frame 运动、其他 site sensor 形式、terrain、其他 reset randomization 与 same-layout 原生播放均快速失败。原生证据使用 MotrixSim Core 0.10.1 与 MuJoCo 3.11.0；不声明超出已记录 CPU profile 验收的能力。见[实体场景执行](entity-scenes.md)。

Motrix profile 还接受场景级、world-referenced 限定 site 的 `FrameLinVel` / `FrameAngVel` fragment，并审计原生类型、site 对象、world-reference 身份、维度以及完整的 parent/local-pose site 身份。结果行按不可变 variant assignment gather；`FrameLinVel` 返回世界系 site 点速度，`FrameAngVel` 返回世界系角速度。entity 内 frame 运动与引用形式均快速失败。

Motrix profile 还接受场景级、world-referenced 限定 body 的 `FrameLinVel` / `FrameAngVel` fragment，并审计原生类型、body 对象、world-reference 身份和维度。结果行按不可变 variant assignment gather；`FrameLinVel` 返回 inertial body-frame origin 处的世界系速度，`FrameAngVel` 返回世界系角速度。entity 内 body 运动与引用形式均快速失败。

Motrix profile 还接受 entity 内、未引用参考系的 site `velocimeter` 与 `gyro` 声明。其原生 `FrameLinVel(local)` 与 `FrameAngVel(local)` 值通过审计后的 variant context gather；accelerometer、引用参考系 site 运动与 entity 内 frame-motion 声明均快速失败。

Newton 支持 CUDA graph 显式开启：`NewtonBackend(..., use_cuda_graph=True)` 或 `create_backend(..., newton_use_cuda_graph=True)`。只有冷路径容量校准重建最终固定地址 state 之后才会捕获 graph，并按 Newton 输入/输出 state 的奇偶交替各捕获一张。捕获要求 CUDA 设备、12.4 及以上驱动和已启用的 CUDA mempool；否则 Newton 会发出带原因的 `RuntimeWarning` 并保持 eager 执行。捕获失败同样回退 eager。state reset 与已注册的 pre-step control callback 保持 eager；无 callback 的物理步按当前 state 奇偶选择并 replay graph。

Newton 播放在只安装单个 `newton` extra 时通过 `ViewerGL`（`pyglet>=2.1.6,<3` 与 `imgui-bundle>=1.92.0`）原生渲染：`record` 离屏渲染，`interactive` 打开窗口 viewer，`auto` 根据显示可用性选择。运行时不完整时，`record` 回退到离线 MuJoCo snapshot 管线，`interactive` 以可操作错误快速失败。无头离屏 GL 需要 EGL（`PYOPENGL_PLATFORM=egl`），或在 Wayland 下使用 GLX。

Drake 的 portable-entity profile 覆盖无 variant 场景和实际使用的同布局 fixed variants。Fixed variants 为每个使用中的 variant 使用一个 DrakeUni runtime，显式 scatter/gather 公开行，严格比较共享 control/sensor 元数据，并强制读回原生 body/primitive 属性；逐环境 playback 不可变。kinematic mirror、control 恢复、mesh/convex 身份、未支持 layout 与 primitive 均快速失败。需要高于已发布 `0.1.0`、同时包含多实体布局排序和原生属性读回的 DrakeUni 构建。有界契约与原生证据见[实体场景执行](entity-scenes.md)。

## 语义清单

下表由 `src/unisim/support.py` 中的 `get_adapter_capabilities()` 生成；运行 `uv run scripts/diagnostics/check_support.py --check-docs` 校验，或用 `--write-docs` 同步生成两种语言。这些是源码审查声明，不是真实运行验证。`exact` 仅针对所述子集；`approximate` 要求逐项授权，`unsupported` 拒绝对应请求，`unknown` 不承诺支持。`*` 表示必须满足声明中的配置条件；原因、条件和固定版本源码证据可从公共报告查询。当前只声明 default profile；未知 profile 保持未知。

<!-- semantic-inventory:start -->
| Feature | mujoco | motrix | drake | mjwarp | newton | superdex | genesis | isaacgym | isaacsim |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `asset.mjcf` | exact | exact | exact | exact | exact | exact | exact | exact | exact |
| `asset.urdf` | unknown | unknown | unknown | unknown | unknown | unknown | unknown | unsupported | unsupported |
| `entity.single_articulation` | exact | exact | exact | exact | exact | exact | exact | exact | exact |
| `entity.multiple` | exact* | exact* | exact* | exact* | exact* | exact* | exact* | exact* | exact* |
| `root.free` | exact | exact | exact | exact | exact | exact | exact | exact | exact |
| `root.fixed` | exact | exact | exact | exact | exact* | exact | exact* | exact* | exact* |
| `joint.hinge` | exact | exact | exact | exact | exact | exact | exact | exact | exact |
| `joint.slide` | exact | exact | exact | exact | exact | exact | exact | exact | exact |
| `joint.ball` | exact | unknown | unknown | exact | unknown | unsupported | unknown | unknown | unknown |
| `actuator.motor` | exact | unknown | unknown | exact | exact | exact | unsupported | unsupported | unsupported |
| `actuator.position` | exact | exact | unknown | exact | unknown | unknown | exact | exact | exact |
| `collision.rigid` | exact | exact | exact | exact | exact | approximate* | exact | exact | exact |
| `collision.self` | exact* | unknown | unknown | exact* | unknown | unknown | unknown | unsupported | exact* |
| `contact.query` | exact | unknown | unknown | exact | exact* | approximate* | approximate* | approximate* | approximate* |
| `terrain.heightfield` | exact | exact | unknown | exact | unknown | unsupported | unknown | unknown | unknown |
| `sensor.imu` | exact | unknown | unknown | exact | approximate | approximate | approximate | unsupported | unsupported |
| `sensor.gyro` | unknown | unknown | unknown | unknown | unknown | unknown | unknown | approximate | approximate |
| `reset.state` | exact | exact | exact | exact | exact | exact | exact | exact | exact |
| `dr.interval.body_force` | unknown | unknown | unknown | unknown | unknown | unknown | unknown | unknown | unknown |
| `state.final_refresh` | exact* | unknown | unknown | exact | unknown | unknown | unknown | unknown | unknown |
| `state.callback_refresh` | exact* | unknown | unknown | exact | unknown | unknown | unknown | unknown | exact* |
| `variant.same_layout` | unknown | unknown | unknown | unknown | unknown | unknown | unknown | unknown | unknown |
<!-- semantic-inventory:end -->

DR、播放、body wrench 和 fixed-variant 能力仍由既有实例 API 提供权威信息。静态清单有意将依赖这些来源的项目保留为 unknown；`backend.get_capabilities()` 聚合实例权威声明。多个逻辑实体分区不代表任意多 articulation 组合。URDF 调研和未合并分支不构成当前支持。IsaacSim legacy 路径预留的零接触缓冲区既不代表有效接触查询，也不代表没有物理接触；映射场景把具名 geom-pair `contact data="force" reduce="netforce"` 声明路由到专用 PhysX 碰撞对力槽位，把 wildcard body-net force（省略 `geom2`）与 body-net `data="found"` 标志路由到批量逐 entity PhysX contact view；legacy model-file 场景的一切 contact 声明均快速失败。Isaac worker 的传感器支持 gyro 重建，但拒绝 accelerometer。

[能力设计决策](adr-capabilities.md) 定义证据匹配和快照生命周期。上方安装表中的 `available` 始终不能用于判断任务兼容性。
