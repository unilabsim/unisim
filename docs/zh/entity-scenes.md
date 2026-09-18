# 实体场景执行

[English](../en/entity-scenes.md) | [中文](entity-scenes.md)

本文说明各 adapter 如何执行[实体决策](adr-entities.md)定义的公共实体、不可变身份和局部 reset 契约。冷路径 authoring/编译边界见[可移植 MJCF 决策](adr-portable-mjcf.md)。MuJoCo 系 adapter 编译并运行完整场景；Isaac adapter 保留独立 worker 和原生执行。

旧 MuJoCo/MJWarp 编译场景也在冷路径使用相同的 `CompiledModelIndex` audit。它记录原生 body 分区、root、joint qpos/qvel 地址、mocap 地址及 actuator transmission/control 列，不重命名匿名对象，也不把 tendon/site/root transmission 假装成标量 joint actuator。旧 whole-model API 保留源语义；只有分区交叉校验通过时才暴露受限实体布局。

## 源准备与身份

公共 portable MJCF compiler 校验源、默认值、名称、引用资源和同布局 variants，产出 expanded MJCF、冻结的 `CompiledSceneLayout`、source/intent report 与内容身份。独立 worker 资产写入编译器派生的显式 body 惯性参数及关节限位。geometry 名称、body 归属、源 contact mask、源摩擦与 sphere 半径也随 cold-path 意图表传输，并保留每个 body 的源 geom 顺序。单位 gear position drive 意图校验并保存为独立表后，从导出 XML 删除 actuator；原生 MJCF importer 无法安全消费 MuJoCo canonical general actuator 拼写。此 profile 明确拒绝源被动关节 damping/弹簧、activation state、不支持的 transmission 和非标量关节。编译后的逐环境 actuator 控制限位用于 step target 及初始/完整 reset control；未启用限位时不按存储的零范围夹紧。

跨 entity sensor 声明位于场景级、仅含 sensor 的 `fragment_files` 输入，fragment 字节属于 portable identity。Fragment 可声明有序 geom-pair contact 形式，或使用最终限定名的 world-referenced body `FramePos` / `FrameQuat` 形式；MuJoCo 将其保留在 expanded MJCF 中，mapped IsaacSim 通过 `ContactSensor` filter 消费 force/netforce contact 形式，Motrix 消费经过 audit 的两种 contact 形式与 body-pose 形式。

生成文件名使用安全的内部 USD 标识，不定义公共实体身份。编译后编辑从当前 spec 序列化，避免写出旧编译结果。宿主持有生成的完整场景及独立源，直到 worker 关闭。Worker 返回的实体名称、完整 assignment 和实际实例质量均与编译意图核对。Worker audit 还确认原生拓扑、drive、惯量以及选中的 visual sphere 半径采用情况。Echo 本身不是独立资产身份证据。

Newton 消费 portable compiler 生成的完整逐 variant MJCF，而不使用同构模板复制。它把每个源导入独立的公开 `ModelBuilder`，在每次 `begin_world()` 世界中选择对应源，并为每个物理实体绑定一个公开 `ArticulationView`。冷路径 audit 在构造 solver 前，将每个实际世界的重力、质量、COM、惯性张量、shape 类型和尺寸与选中源逐一比较。此有边界 profile 覆盖每世界多个 articulation 与多个 free root、固定根、被动实体、静态刚体、同布局同 shape 类型的异构 variants 与力响应、局部实体 reset/playback，以及带逐世界归因的具名 geom-pair found sensor。kinematic mirror 与混合 shape 类型 assignment 均快速失败。

Portable Newton 局部 reset 保留无关状态与 control，清空选中实体 control，并对 `restore_default_controls` 快速失败；不声明 keyframe/default control 恢复。

宿主发布独立 nq/nv/nu 和实体根布局。公共 root state 使用 link 原点位置/线速度、wxyz 四元数和世界系角速度；clone offset 由 adapter 恰好消除一次。完整广义 qvel 保留 body 系角速度。被动关节贡献状态，但不隐式增加 action 列。

## Reset 与控制

全部实体 patch 在物化或写原生状态前校验。准备好的行和显式 mask 经同一版本化 reset 命令提交。新实体入口的完整 qpos/qvel 写也归一为相同 patch 提交。完整 `reset()` 恢复选中环境的 variant 默认值和独立 keyframe control；control 不必等于关节位置。其它环境和未选实体保留状态与 targets。

IsaacGym 将 indexed root/DoF 提交累积到下一物理步，避免第二次 indexed setter 覆盖较早 reset。原生 COM 速度在公共 link 原点边界两侧转换。Root 和 joint state 立即可用；初始化或受影响 reset 后的 articulation descendant body/sensor state 在下一 step 前明确不可用。宿主拒绝相关读取，不把旧值当作当前状态。旧 model-file 路径行为不同：PhysX 不推进物理就无法刷新 link pose，因此在 INIT keyframe 或 `set_state` 之后、首个物理步之前，worker 将精确的 MJCF 前向运动学叠加到刚写入的环境上——由有效广义状态计算位置、姿态和 link 原点线速度/世界角速度——并同时清除其陈旧的 contact force 行。宿主把运动学树扫描进 INIT payload（fixed variants 下逐 variant），与公开 body/joint 布局不一致时 fail-closed。legacy reset 输入遵循 canonical 广义速度约定（世界系 link 原点线速度、body 系角速度）；历史 COM 速度投影仅保留在对外发布的 legacy 输出缓冲上。

无法恢复的原生提交失败会设置 worker fault 标记，宿主拒绝后续状态和 step 使用。提交前校验失败保留会话。共享内存槽在 attach 前校验 shape/dtype，包括零宽动作或状态布局。

## 检查与回放

Worker 必须提供版本化配置报告，宿主校验后才接受。实体 assignment、body mass 与 geometry 使用作用域记录，区分源意图与实例读回；源值不替代缺失的运行值。mapped IsaacSim 场景中，INIT 会校验原生 mass、COM 与 geometry 记录；`get_body_mass()` 再将这些 worker 原生快照 scatter 到冻结的公共 body 顺序，返回独立的 `(num_envs, nbody)` 表。未被实体拥有的公共行保留编译源规范值；body mass 只支持读回，因为 reset 时修改不被支持。`get_body_ipos()` 返回形状为 `(nbody, 3)` 的独立编译源规范默认值；`get_body_ipos(env_ids=...)` 返回形状为 `(len(env_ids), nbody, 3)` 的 worker 原生物化行，并保留空选择、重复项与乱序选择。公开 geometry 按冻结的 entity/geom 记录排序：`get_geom_names()` 返回具名地址，`get_geom_body_ids()` 返回公共归属 body ID，`get_geom_contact_masks()` 将原生 collider 启用状态规范化为一对相同的 contype/conaffinity 行，`get_geom_friction()` 返回形状为 `(num_envs, ngeom, 3)` 的独立 worker 原生 PhysX material 行，格式为 `[static, dynamic, 0]`。初始 material 随不可变 variant assignment 变化；局部 reset 可以替换当前 material 行，普通状态 reset 不会隐式恢复它们。collision 状态仍由 role 决定且不可变。worker 从实际 spawn 的 collision prim 读取身份与 material，并拒绝数量、名称、归属、mask 或 material 漂移。独立的 visual `Sphere.radius` audit 仍是有边界 sphere profile 检查，不是尺寸 API。

mapped `set_state()` 使用既有 `ResetRandomizationPayload` 支持局部环境的 `geom_friction` 写入。friction 是绝对 dense 公共布局表，必须有限非负，满足 `static == dynamic`，第三列精确为零。该命令复用 `RESET_ENTITIES`，通过映射后的公共 PhysX material view 写入，写后读回原生值，校验选中行与未触碰行，刷新宿主当前值 getter；原生写入/读回失败会使 worker fault。`body_mass` 在 capability 边界 fail closed：当前 IsaacSim GPU pipeline profile 中，公开 mass 写入可能更新读回值，但 active solver 会保留旧 mass 直到后续非零 solver step，而 reset 不能隐式推进物理。gravity 保持不支持，因为它是仿真全局量而不是局部环境状态。body mass、惯性与 COM 的 reset 修改保持不支持；源码中存在 `set_coms()` 不构成 COM 证据。joint damping/frictionloss 保持不支持，因为源 passive damping 与导入 drive damping 没有一个可辩护的公共当前边界；actuator `kp`/`kd` 保持不支持，因为现有 getter 返回不可变 INIT/源值而非 reset 后当前值。几何维度、其他 contact 参数以及所有其他 reset 项同样 fail closed。版本化 import report 描述 materialization，不能当作 reset 后的当前值读回。回放返回选中环境的完整场景源，保留 robot、object、table 和 mirror。原生渲染仍由 worker 拥有；本切片尚未提供 mapped worker 的 physics snapshot 导出。

当前原生相机 profile 使用既有跟踪行为，拍摄环境 0 的第一个实体。录制只支持配置 `cam_distance`、`cam_elevation` 和 `cam_azimuth`。非默认的 `cam_lookat`、`cam_tracking`、`cam_tracking_env_idx`、`cam_tracking_extra_envs` 或 `cam_fov` 在访问 worker 前抛出 `NotImplementedError`，重复初始化 renderer 时也会校验。默认 `CameraCfg` 保留既有原生视图，不会选择 MuJoCo 网格相机。交互 viewer 也拒绝自定义球面偏移，其视图由原生 viewer 控制。可以返回任意选中环境的完整场景源，不代表原生相机可以选择该环境。

IsaacSim 将不可变 same-drive assignment 映射为 K 个唯一原型，并把选中的原型复制到每个精确环境路径。assignment 在 construction 固定，reset 不重采样。单 body rigid view、固定根模式、环境 view 行映射、mass/COM/惯量以及选中的 visual sphere 半径均独立 audit；拓扑、drive 与源格式限制仍 fail closed。共用宿主只启用已说明的 MJCF 标量关节 profile，不代表 URDF 或全部 PhysX 资产特性。

IsaacSim 缓存两个不可变冷路径 USD stage。Raw cache 在 `~/.cache/unisim/isaacsim/raw-usd` 下保存 expanded MJCF 到 raw USD 的转换输出；可将 `UNISIM_ISAACSIM_RAW_USD_CACHE` 设为绝对路径，或使用只读兼容回退 `UNILAB_ISAACSIM_RAW_USD_CACHE`。Raw identity 在 portable scene content identity 之上追加 expanded source digest、importer 参数、IsaacSim/IsaacLab/MJCF importer/Python 版本、实体和 variant。Role cache 在 `~/.cache/unisim/isaacsim/role-usd` 下保存独立 bake 的 USD；可将 `UNISIM_ISAACSIM_ROLE_USD_CACHE` 设为绝对路径，或使用只读兼容回退 `UNILAB_ISAACSIM_ROLE_USD_CACHE`。空值或 `0`、`false`、`no`、`off`、`none`、`disabled`（不区分大小写）会独立禁用对应缓存。Role identity 在 raw identity 与不可变 raw artifact 指纹之上追加 source/role 实体、variant、kind、root mode、collision/mirror 设置和 bake 参数。命中会校验完整 manifest、文件清单、大小和 SHA-256；转换与 bake 输出先写入 staging 再原子发布，格式错误、中断或损坏条目按 miss 处理并替换。Role miss 会先复制完整 raw artifact 再 bake。Role hit 会重新读取 variant 元数据、native body path、articulation/rigid 拓扑、mobility、gravity、collision 和已禁用的 converter drive，然后才执行原生 materialization、prim/view 构建、参数读回、capability 检查和 effective-report 校验。禁用 role cache 时仍会为每个 role 生成独立临时副本，因此 role 编辑不会修改缓存 raw USD 或另一个 role。

映射 IsaacSim 场景通过 IsaacLab `ContactSensor` filter 服务 `<contact data="force" reduce="netforce"/>`。数值单位为牛顿，是世界系 3 向量，表示 `geom2` 目标 body 作用在 `geom1` 来源 body 上的 filtered normal contact force；IsaacLab 已对接触点求和。`get_sensor_data()` 返回最后一个已完成物理子步，不在子步间平均；指定碰撞对无接触时返回零，reset 清理陈旧行。body-net `data="found"` 和未支持的 reduction 均快速失败。legacy 预留 body-net 槽位绝不作为碰撞对查询暴露。SDK-free 测试固定声明、wire、行映射和 reset 语义；gated real-worker 套件检查 1 kg 与 2 kg 的静态支持力。

Mapped IsaacSim 还支持作用于原生 body COM 的世界系 `body_force` 与 `body_torque` 区间 wrench。多次提交在下一次 step 前累加，作用于请求的每个子步，并在随后消费。Entity reset 清理选中实体 body，完整 reset 清理所有暂存 body 行。映射场景也接受既有宿主 `set_pre_step_control()` callback：宿主把一个公开子步对应为一次 worker STEP，并在每次 callback 前刷新共享状态。callback 的 control 与动态 wrench 每个子步从头重算，与暂存区间 wrench 叠加，并在 step 调用结束时消费。这是正确性宿主循环，不是 #152 跟踪的 device-resident controller 或性能路径；legacy model-file callback 仍不支持。

## Drake 有边界 portable profile

Drake 消费公共 portable compiler 产出的 expanded MJCF，并只在 DrakeUni 公开 model 信息与 body 查询和冻结的 `CompiledSceneLayout` 完全一致时接受布局：维度、全局 body 名称/ID、每个 root 与标量 joint qpos/qvel 映射、joint 全量消费、actuator 名称顺序及目标地址都必须精确绑定。固定 root 通过公开 body-state 查询读取；浮动 root 使用广义状态快照。被动关节贡献状态与速度，但不隐式增加控制列。

局部 entity reset 先快照完整场景状态，应用已校验 patch，再通过 DrakeUni 公开 reset 提交完整、一致的行为。未选中的实体与环境保留状态。原生 reset 失败会使 backend 进入 faulted。`restore_default_controls=True` 会在变更前被拒绝，因为 DrakeUni reset 尚未提供显式 control 恢复契约。entity 模式同样拒绝隐式单 root base/body-frame 辅助接口。

首个 Drake profile 只支持无 variant、固定或浮动 root 的 MJCF 物理实体、被动标量关节以及固定 rigid 静态实体。固定 variants 与 kinematic mirrors 会在加载 DrakeUni 或物化 portable model 前快速失败。Drake 原生验收是支持声明的必要条件；缺少 Drake native batch extension 时可选测试会跳过。DrakeUni 必须包含多实体布局排序修复（当前发布的 `drake-uni==0.1.0` 尚不包含该前置修复）。

## Genesis 有边界 portable profile

Genesis 以公共 portable compiler 作为布局、身份、来源与来源关系的权威。适配器通过公共 loader 归一化每个源，序列化为 Genesis 独立持有的 MJCF 输入，并把每个公开实体分别加入同一个原生 `Scene`。物化时，只有原生状态维度、实际 link 名称、原生惯量矩阵、joint 地址、actuator 顺序和 PD 增益与独立编译源一致才接受实体；绑定绝不假设 Genesis link 顺序。公开 qpos/qvel、body 和 actuator 列随后 scatter 到所属实体，被动实体贡献状态但不产生控制列。

首个 profile 覆盖固定 root articulation、浮动被动 articulation、浮动异构 rigid object 以及固定 rigid 静态实体。异构 variants 仅支持单 link rigid 实体。由于 Genesis 1.3.3 会自行分配 MJCF morph 可迭代对象，构造只接受与该原生行为一致的精确公共 balanced mapping——K 个 variant 在 N 个环境中按连续块分配，前余数个块各多获得一个环境。N5/K2 对应 `[0,0,0,1,1]`；`[1,1,0,1,0]` 会在构造 Genesis scene 前被拒绝。该 owner 侧映射校验独立于 Genesis 私有 solver 辅助函数。

局部 `set_state()` 与 `reset_entities()` 只把选中行提交到所属原生实体，保留未选中的实体与环境。部分原生状态提交失败会使 backend 进入 faulted。Portable mode 不声明 reset randomization，并拒绝 mirrors、kinematic 实体、跨实体 sensor fragment、源 sensor 映射、reset 时 body-force 映射、`restore_default_controls=True`，以及 legacy 单主实体 DoF 视图。固定 root 的世界放置通过公共 MJCF morph position/quaternion 参数传递，因为该导入路径中原生 Genesis 会忽略归一化固定 root body position。

Portable geometry 暴露冻结的限定名称、公开 ID、归属 body ID、有边界的 Genesis 原生 geometry 尺寸、contact mask、摩擦系数与 contact solver 参数。构造会按名称、归属 link、精确 active environment 行以及已审查的源 AABB 边界审计每个源 variant 的具名原生 visual instance；当前 visual 身份仅支持 sphere 与 box，不支持的 geometry 类型和模糊或缺失的原生身份都会快速失败。Geometry 尺寸从这些已审计的原生 AABB 派生：sphere 暴露原生半径，box 暴露原生 half-extent，并按冻结公开顺序聚合；非一致 fixed-variant 尺寸快速失败。`get_geom_contact_masks()`、`get_geom_friction()`、`get_geom_solref()` 与 `get_geom_solimp()` 只按名称、归属 link 与 active environment 行把完全匹配的 collision instance 聚合到冻结公开顺序。mask 是 Genesis 重新编码后的原生值，不是 MuJoCo 表的回显；摩擦列按 Genesis 公开原生系数顺序 `[sliding, torsional, rolling]`，不是 static/dynamic material 表。Genesis 的原生七值 solver 表在公共边界拆分为 MuJoCo 兼容的 `solref = [timeconst, dampratio]` 与 `solimp = [dmin, dmax, width, mid, power]`；因此 importer 预处理或默认值以实际原生值报告，而不是回显源 XML。这些是构造期属性读取，不声明 reset 修改或 solver 功能启用。缺失或模糊的 collision 身份，以及非一致 fixed-variant mask/摩擦/solver 参数，均快速失败。其他 geometry 属性仍不支持。主源的原生 link 惯量矩阵会对照编译出的对角惯量和姿态，但不暴露任意惯量读回。异构惯量记录加 free-body 响应提供有边界原生证据；由于验收 variants 的质量、COM 与半径也不同，该响应并不能单独隔离惯量影响。

原生 variant 质量通过公共 Genesis mass getter 读取并 scatter 到冻结公开 body 行；带选中环境的 `get_body_ipos(env_ids=...)` 暴露逐 variant COM 行，而无参数 `get_body_ipos()` 仍返回规范的 `(nbody, 3)` 默认表。这些是读回边界，不是 reset 修改能力。

Portable DOF 读回通过已经审计的 joint/native qvel 地址映射暴露 Genesis 公开的原生 `get_dofs_damping()`、`get_dofs_frictionloss()` 与 `get_dofs_armature()` 表。构造期捕获所有 active-environment 行，同一个 fixed variant 内的非一致行会被拒绝；只有所有 fixed variants 一致时，公开 DOF getter 才返回一份分离的 `(nv,)` 表。被动关节 damping、friction loss 与 armature 仍区别于导入的 position-drive `kp`/`kv`；这些是构造期读取，不声明 reset 修改、恢复或随机化。

## Motrix 有边界 portable profile

Motrix 通过公开 `msd.from_file()` 与 `msd.build()` 导入公共 compiler 生成的完整 expanded 多 root MJCF。构造时，只有实际 Motrix 元数据与冻结的 `CompiledSceneLayout` 一致才接受 entity mode：qpos/qvel/actuator 维度、原生 link 名称、浮动 root 与标量 joint 状态地址、完整广义状态顺序、actuator 名称/顺序/目标以及 geom 名称/顺序都必须一致。Variant context 还会审计原生 body/geom 顺序、公开 control 与 joint limit、actuator 增益和 geom 摩擦，捕获实际原生 geometry 尺寸，并读回构造 mass/COM。公开 body 与 geom ID 通过这些审计后的原生映射 scatter，而不是假设 Motrix link 或 geom 顺序。

首个 profile 覆盖无 variant 与不可变 same-layout fixed-variant 的固定/浮动物理实体、被动标量关节和固定 rigid 静态实体。Motrix 为每个被使用的 variant 创建一个原生 model/data context，并按不可变 assignment 显式 scatter/gather 公开 qpos/qvel、control、局部 reset 行、step 执行和 link pose/velocity 行。运行前变更发生前，每个 context 会捕获原生构造 control；若配置 `default_keyframe_name`，则捕获唯一同名原生默认 keyframe 的 `ctrl` 记录，并按已审计的原生 control limit 截断。Variant geometry、mass、COM 与惯量可以不同；`get_body_mass()` 暴露逐环境原生行，带选中环境的 `get_body_ipos()` 暴露原生 COM 行，审计后的原生 geometry 记录保留实际逐源尺寸。Motrix 不提供惯量读回，因此原生验收对被动标量 joint 施加相同 torque 并验证不同的有效响应，而不是假装存在 getter。非一致的公开 control/joint limit、actuator 增益或 geom 摩擦会快速失败，因为公开 API 暴露规范标量表。`apply_body_force()` 将世界坐标 public body 行映射到每个拥有它的 variant context：提交会为即将到来的原生 step 加性累积，并在该 step 消耗；局部 root/joint reset 只取消受影响 entity 的 body 行，无关 pending force 保持不变。Body torque 仍不支持。

当 `add_body_sensors=True` 时，Motrix 会在 variant 物化前把 `base_name` 解析到冻结的公开 body 布局：限定名 `entity/body` 必须精确匹配，本地名必须唯一匹配一个公开 body。随后每个原生 variant 会为所有公开 body 生成 Motrix frame-position 与 frame-quaternion sensor。构造要求每个公开 body 恰好两个 sensor、各 variant 名称完全一致，并校验原生行数和维度。公开 sensor getter 按不可变 variant assignment 从所属 context gather 行；batch 读取按请求顺序展平并拼接每个 sensor。`get_sensor_data()` 保持原生 xyzw 四元数，`get_body_quat_b()` 转换为公开 wxyz 约定。局部 reset 会刷新派生 frame sensor，但不会提交无关广义状态行。

Authored source 与场景级 fragment frame sensor 仅接受 public body 上的 world-referenced Motrix `FramePos` / `FrameQuat`。Entity-owned source 名称保留 owner 前缀且目标必须属于同一 entity；fragment 名称不带前缀，可指向任意 public body（包括其他 entity）。公共 compiler 会跨 variant 冻结其公开名称、类型、维度和对象引用；Motrix 物化额外要求完整的具名 sensor 集合、原生 frame sensor 类型/对象/引用身份一致、原生行维度精确，以及 fixed variant 名称完全一致。这些 `FramePos` 读取暴露世界系位置，`FrameQuat` 保持原生 xyzw 输出。它们可与生成的跟踪 sensor 混用，并按不可变 variant context gather。

场景级 geom-pair contact fragment 仅使用公共 `data="force" reduce="netforce"` 与 `data="found" num="1"` 形式。Motrix 在构造前 audit 每个 native contact sensor 的精确 geom pair、reduction mode 和 force/found report 标志，要求原生维度精确且 fixed variant 名称一致，并返回已完成 step 后留在原生 sensor storage 中的值。不支持的 contact reduction 与 site sensor 均快速失败。

局部 `reset_entities()` 行通过 `SceneData[DisjointIndices]` 一致提交，不调用整场景 reset，并保留未提及通道、无关实体、无关环境与当前 control。`restore_default_controls=True` 时，只把被选中 entity root/joint patch 影响的 control 恢复为该 variant 冷路径捕获的原生默认值；无关 control 列与行保持不变。部分原生状态提交会使 backend 进入 faulted。其他 fixed-variant layout、kinematic mirrors、其他 source sensor 形式、site sensor、不支持的 contact 形式、生成 terrain、reset randomization、portable site Jacobian、fixed-variant 原生播放/渲染与非一致公开 geometry 尺寸读取均快速失败。固定 link 的质量可能读为零，因为公开 mass/COM 是有效原生读回，不是源 inertial 元素的回显。

## Adapter profiles

| Adapter | 当前 profile | 绑定与 reset 边界 |
| --- | --- | --- |
| MuJoCo | 支持含固定/浮动/kinematic 实体、镜像、被动关节和 same-layout variants 的 MJCF 源。 | 一个编译后的 `mjbatch` 场景使用冻结公共地址；局部 reset 只 scatter 受影响行并保留无关通道。 |
| Motrix | 支持无 variant 与不可变 same-layout fixed-variant、固定/浮动 root 的 MJCF 物理实体、被动标量关节、固定 rigid 静态实体、生成的 body 系跟踪位姿 sensor 和 authored 世界系 body 位姿 sensor。 | 每个被使用的 variant 创建一个原生 Motrix model/data context；审计后的公开 state/control/sensor 行被 scatter/gather，局部 reset 保留无关状态，不支持的语义快速失败。 |
| MJWarp | 在 MuJoCo 组合 profile 上增加 CUDA 逐世界 variant 字段和具名编译几何。 | 单个 model/data runtime 原地上传选中值，恢复持久通道并 forward 主 Data；不声明存在选择性原生 forward。 |
| Drake | 支持无 variant、固定/浮动 root 的 MJCF 物理实体、被动关节和固定 rigid 静态实体。 | 一个 expanded portable model 在冷路径对照公开布局元数据审计；局部 reset 提交一致完整行，不支持 variants、mirrors 与 control 恢复时快速失败。 |
| Newton | 有边界 portable MJCF 实体，支持固定/浮动根、被动/静态 body、同布局同 shape 类型 variants 与具名 found contact。 | 独立逐 variant builder 被分配到显式世界；公开逐实体 articulation view 与原生身份 audit 隔离局部状态 reset 和接触世界归因，不支持的 mirror 与混合 shape 类型快速失败。 |
| Genesis | 有边界 CPU profile 的 portable MJCF 固定/浮动/被动/静态实体与单 link rigid 异构 variants。 | 独立原生实体按审计后的公开名称与地址绑定；局部 state/reset 行保留无关状态，审计后的原生 sphere/box 尺寸与完全一致的 contact mask 按公开顺序聚合，任何非 balanced assignment 或不支持的 sensor/DR 映射快速失败。 |
| IsaacGym | 支持独立 MJCF 实体、标量关节、position drive、rigid 镜像和不可变任意 assignment。 | 审计查询到的 actor/body/DoF 索引；indexed 写入在下一步前合并，后代 body 读取显式暴露新鲜度边界。 |
| IsaacSim | 支持 articulation/rigid view、标量关节、不可变同 drive K 原型 assignment、单 body 刚体、geometry 读回、局部 friction 修改、映射碰撞对力传感器和暂存世界系 body wrench。 | 审计 prim/view、assignment、body/joint 映射、原生 body/geometry 属性与选中 sphere 半径；局部 material 写读回原生当前行，保留未提及通道/环境，提交后的失败会使 worker 进入 faulted。 |

## 验证

`tests/contract/test_worker_scene_native.py` 通过 `UNISIM_TEST_ISAACGYM_SCENE=1` 或 `UNISIM_TEST_ISAACSIM_SCENE=1` 启用真实 factory-to-worker 验收。测试覆盖两个后端的 N5/K2 非 round-robin 身份、nq/nv/nu、局部 reset 隔离与持久性、与 qpos 不同的 keyframe control、完整场景回放及版本化导入报告。各 worker 测试还覆盖独立原生质量/COM/惯量、拓扑、镜像、被动 articulation 和过滤接触力；IsaacSim gate 还会读取原生 geometry 身份、collision 状态与 PhysX material——包括同一 body 上的多个 collision prim——并从已 spawn 的 visual USD 读取选中原生 sphere 半径，同时覆盖 force、torque、消费、selected-reset wrench、逐子步 callback 状态/wrench 组合行为、带原生读回与物理响应的局部 friction 修改、reset 时 body-mass fail-closed 行为，以及带原生读回和缓存文件不可变校验的 raw/role USD 冷/热命中。最终 #72 测试用同一个 MJCF-only robot/object/table/mirror 操作场景，在 MuJoCo 与 gated IsaacSim 中通过公共构造、布局、属性、接触、reset、逐子步控制、wrench 与 playback 检查，并记录 runtime/GPU 证据且排除 #133 camera/RGB。IsaacSim 测试被跳过不构成原生证据。

Newton 原生 gate 是真实 CUDA 的 `tests/adapters/newton/test_multi_entity_feasibility.py`。其 N5/K2 `[1,1,0,1,0]` assignment 覆盖受控固定根 robot、被动浮动 object 和静态 table；测试校验逐世界 mass/COM/惯量/shape 身份及其持久性、源默认值、object-only 与 joint-only 局部 reset 隔离、无关 control 保留、跨静态零宽 view 的完整选中快照、非单位姿态与偏移 COM 的坐标转换、被动 action 宽度、actuator 力响应、同几何下随 variant mass/inertia 变化的力响应、source-order 地址置换、独立 N1 与 N5 parity、backend 生命周期内的逐 variant playback 路径、close 后生成源清理、真实接触世界隔离，并在构造 solver 前确定性拒绝混合 shape 类型 variants 与 kinematic mirror。这些是上述有边界 profile 的验收证据，不代表任意拓扑、renderer 等价、keyframe control 恢复或吞吐优化。

已记录的原生证据使用 Newton 1.5.1、Warp 1.16.0 与 MuJoCo 3.11.0，GPU 为 NVIDIA GeForce RTX 4090。

Genesis portable-entity 验收是真实原生 CPU 测试 `tests/adapters/genesis/test_portable_entities.py`。其 N5/K2 `[0,0,0,1,1]` assignment 覆盖受控固定根 robot、浮动被动 articulation、异构 rigid object 与固定静态 table；测试校验公开布局维度与 actuator 宽度、通过实际原生 link 名称/ID 绑定、原生 variant 质量 `[0.5,0.5,0.5,1.5,1.5]`、公开 geometry 名称/ID/body 归属、实际原生 geometry 尺寸、按公开顺序聚合的实际原生 contact mask、原生 DOF damping/friction-loss/armature 读回、active 原生 visual instance 及其源 AABB、局部 state 隔离、局部 reset 隔离、源 variant 惯量记录、joint 控制的物理响应以及异构 free-body 旋转响应。测试还通过公开 size getter 拒绝故意非一致的 object variant 半径。SDK-free owner 测试覆盖缺失/variant 非一致 mask 与同 variant 非一致 active 行 DOF 的拒绝。记录的运行时为 Genesis 1.3.3、Torch 2.14.0+cpu 与 Quadrants 1.3.0；这是 CPU 证据，不构成 GPU 声明。Genesis 测试被跳过不构成原生证据。

Motrix portable-entity 验收是真实原生测试 `tests/adapters/motrix/test_portable_entities.py`。它覆盖重复本地名称、两个浮动 root、被动关节物理响应、公开/原生 body 与 geom 映射、subtree ID、原生 mass/COM 与 geometry 尺寸读回、保留 control 的局部 reset 隔离、被动 joint reset、生成的 frame-position/quaternion sensor 读取与局部 reset 行隔离、与生成 sensor 混用的 authored 世界系 body 位姿 sensor、N5/K2 `[1,1,0,1,0]` same-layout assignment、不同有效标量 joint 惯量响应和按 assignment 路由的生成/source sensor gather，以及模糊 base、不支持 source sensor 与 fragment 拒绝、不支持 layout 拒绝和组合源清理。记录的本地证据使用 Python 3.13.14、MotrixSim Core 0.8.2、MuJoCo 3.11.0 与 NumPy 2.5.2；Motrix 测试被跳过不构成原生证据。

既有 `model_file` 入口保留冷路径 importer 和源配置，随后将已初始化的原生对象交给显式实体使用的同一个场景执行器。`LegacySlotProjection` 保留历史 root/state/control 缓冲形状与名称，不包含物理循环。两个 worker 均只有一套 step、reset 和 refresh 实现。旧 D 宽动作（含被动列）与合成的 7/6 root 坐标作为显式兼容映射保留，不代表源资产声明了 free joint 或相应 actuator。Gym 历史 COM 线速度输出和世界角速度 root 槽与 canonical link/body 系坐标分别转换。既有地面/importer 策略保留在冷路径，旧 Isaac host 不新增 SDK 依赖。

Drake portable-entity 验收覆盖重复本地名称、两个浮动 root、被动关节可见性与物理响应、局部 reset 隔离、N5 batch 行与独立 N1 runtime 对比、反向声明顺序、不支持 variants/mirrors 的物化前拒绝，以及 close 或冷路径布局不匹配时的清理。Drake 测试被跳过不构成原生证据。
