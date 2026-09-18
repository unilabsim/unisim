# 实体场景执行

[English](../en/entity-scenes.md) | [中文](entity-scenes.md)

本文说明各 adapter 如何执行[实体决策](adr-entities.md)定义的公共实体、不可变身份和局部 reset 契约。冷路径 authoring/编译边界见[可移植 MJCF 决策](adr-portable-mjcf.md)。MuJoCo 系 adapter 编译并运行完整场景；Isaac adapter 保留独立 worker 和原生执行。

旧 MuJoCo/MJWarp 编译场景也在冷路径使用相同的 `CompiledModelIndex` audit。它记录原生 body 分区、root、joint qpos/qvel 地址、mocap 地址及 actuator transmission/control 列，不重命名匿名对象，也不把 tendon/site/root transmission 假装成标量 joint actuator。旧 whole-model API 保留源语义；只有分区交叉校验通过时才暴露受限实体布局。

## 源准备与身份

公共 portable MJCF compiler 校验源、默认值、名称、引用资源和同布局 variants，产出 expanded MJCF、冻结的 `CompiledSceneLayout`、source/intent report 与内容身份。独立 worker 资产写入编译器派生的显式 body 惯性参数及关节限位。geometry 名称、body 归属、源 contact mask、源摩擦与 sphere 半径也随 cold-path 意图表传输，并保留每个 body 的源 geom 顺序。单位 gear position drive 意图校验并保存为独立表后，从导出 XML 删除 actuator；原生 MJCF importer 无法安全消费 MuJoCo canonical general actuator 拼写。此 profile 明确拒绝源被动关节 damping/弹簧、activation state、不支持的 transmission 和非标量关节。编译后的逐环境 actuator 控制限位用于 step target 及初始/完整 reset control；未启用限位时不按存储的零范围夹紧。

跨 entity 碰撞对力声明位于场景级、仅含 sensor 的 `fragment_files` 输入。其内容属于 portable identity；MuJoCo 编译 sensor 与 mapped IsaacSim worker `ContactSensor` filter 消费同一份最终 expanded MJCF。

生成文件名使用安全的内部 USD 标识，不定义公共实体身份。编译后编辑从当前 spec 序列化，避免写出旧编译结果。宿主持有生成的完整场景及独立源，直到 worker 关闭。Worker 返回的实体名称、完整 assignment 和实际实例质量均与编译意图核对。Worker audit 还确认原生拓扑、drive、惯量以及选中的 visual sphere 半径采用情况。Echo 本身不是独立资产身份证据。

宿主发布独立 nq/nv/nu 和实体根布局。公共 root state 使用 link 原点位置/线速度、wxyz 四元数和世界系角速度；clone offset 由 adapter 恰好消除一次。完整广义 qvel 保留 body 系角速度。被动关节贡献状态，但不隐式增加 action 列。

## Reset 与控制

全部实体 patch 在物化或写原生状态前校验。准备好的行和显式 mask 经同一版本化 reset 命令提交。新实体入口的完整 qpos/qvel 写也归一为相同 patch 提交。完整 `reset()` 恢复选中环境的 variant 默认值和独立 keyframe control；control 不必等于关节位置。其它环境和未选实体保留状态与 targets。

IsaacGym 将 indexed root/DoF 提交累积到下一物理步，避免第二次 indexed setter 覆盖较早 reset。原生 COM 速度在公共 link 原点边界两侧转换。Root 和 joint state 立即可用；初始化或受影响 reset 后的 articulation descendant body/sensor state 在下一 step 前明确不可用。宿主拒绝相关读取，不把旧值当作当前状态。旧 model-file 路径行为不同：PhysX 不推进物理就无法刷新 link pose，因此在 INIT keyframe 或 `set_state` 之后、首个物理步之前，worker 将精确的 MJCF 前向运动学叠加到刚写入的环境上——由有效广义状态计算位置、姿态和 link 原点线速度/世界角速度——并同时清除其陈旧的 contact force 行。宿主把运动学树扫描进 INIT payload（fixed variants 下逐 variant），与公开 body/joint 布局不一致时 fail-closed。legacy reset 输入遵循 canonical 广义速度约定（世界系 link 原点线速度、body 系角速度）；历史 COM 速度投影仅保留在对外发布的 legacy 输出缓冲上。

无法恢复的原生提交失败会设置 worker fault 标记，宿主拒绝后续状态和 step 使用。提交前校验失败保留会话。共享内存槽在 attach 前校验 shape/dtype，包括零宽动作或状态布局。

## 检查与回放

Worker 必须提供版本化配置报告，宿主校验后才接受。实体 assignment、body mass 与 geometry 使用作用域记录，区分源意图与实例读回；源值不替代缺失的运行值。mapped IsaacSim 场景中，INIT 会校验原生 mass、COM 与 geometry 记录；`get_body_mass()` 再将这些不可变 worker 原生快照 scatter 到冻结的公共 body 顺序，返回独立的 `(num_envs, nbody)` 表。未被实体拥有的公共行保留编译源规范值。`get_body_ipos()` 返回形状为 `(nbody, 3)` 的独立编译源规范默认值；`get_body_ipos(env_ids=...)` 返回形状为 `(len(env_ids), nbody, 3)` 的 worker 原生物化行，并保留空选择、重复项与乱序选择。公开 geometry 按冻结的 entity/geom 记录排序：`get_geom_names()` 返回具名地址，`get_geom_body_ids()` 返回公共归属 body ID，`get_geom_contact_masks()` 将原生 collider 启用状态规范化为一对相同的 contype/conaffinity 行，`get_geom_friction()` 返回形状为 `(num_envs, ngeom, 3)` 的独立 worker 原生 PhysX material 行，格式为 `[static, dynamic, 0]`。摩擦只随不可变 variant assignment 变化；collision 状态由 role 决定且不可变。worker 从实际 spawn 的 collision prim 读取身份与 material，并拒绝数量、名称、归属、mask 或 material 漂移。独立的 visual `Sphere.radius` audit 仍是有边界 sphere profile 检查，不是尺寸 API。legacy model-file 路径仍快速失败。非 sphere 几何尺寸、geom 尺寸、contact 参数与属性修改仍不支持；接触力与暂存 wrench 见下文。回放返回选中环境的完整场景源，保留 robot、object、table 和 mirror。原生渲染仍由 worker 拥有；本切片尚未提供 mapped worker 的 physics snapshot 导出。

当前原生相机 profile 使用既有跟踪行为，拍摄环境 0 的第一个实体。录制只支持配置 `cam_distance`、`cam_elevation` 和 `cam_azimuth`。非默认的 `cam_lookat`、`cam_tracking`、`cam_tracking_env_idx`、`cam_tracking_extra_envs` 或 `cam_fov` 在访问 worker 前抛出 `NotImplementedError`，重复初始化 renderer 时也会校验。默认 `CameraCfg` 保留既有原生视图，不会选择 MuJoCo 网格相机。交互 viewer 也拒绝自定义球面偏移，其视图由原生 viewer 控制。可以返回任意选中环境的完整场景源，不代表原生相机可以选择该环境。

IsaacSim 将不可变 same-drive assignment 映射为 K 个唯一原型，并把选中的原型复制到每个精确环境路径。assignment 在 construction 固定，reset 不重采样。单 body rigid view、固定根模式、环境 view 行映射、mass/COM/惯量以及选中的 visual sphere 半径均独立 audit；拓扑、drive 与源格式限制仍 fail closed。共用宿主只启用已说明的 MJCF 标量关节 profile，不代表 URDF 或全部 PhysX 资产特性。

IsaacSim 缓存两个不可变冷路径 USD stage。Raw cache 在 `~/.cache/unisim/isaacsim/raw-usd` 下保存 expanded MJCF 到 raw USD 的转换输出；可将 `UNISIM_ISAACSIM_RAW_USD_CACHE` 设为绝对路径，或使用只读兼容回退 `UNILAB_ISAACSIM_RAW_USD_CACHE`。Raw identity 在 portable scene content identity 之上追加 expanded source digest、importer 参数、IsaacSim/IsaacLab/MJCF importer/Python 版本、实体和 variant。Role cache 在 `~/.cache/unisim/isaacsim/role-usd` 下保存独立 bake 的 USD；可将 `UNISIM_ISAACSIM_ROLE_USD_CACHE` 设为绝对路径，或使用只读兼容回退 `UNILAB_ISAACSIM_ROLE_USD_CACHE`。空值或 `0`、`false`、`no`、`off`、`none`、`disabled`（不区分大小写）会独立禁用对应缓存。Role identity 在 raw identity 与不可变 raw artifact 指纹之上追加 source/role 实体、variant、kind、root mode、collision/mirror 设置和 bake 参数。命中会校验完整 manifest、文件清单、大小和 SHA-256；转换与 bake 输出先写入 staging 再原子发布，格式错误、中断或损坏条目按 miss 处理并替换。Role miss 会先复制完整 raw artifact 再 bake。Role hit 会重新读取 variant 元数据、native body path、articulation/rigid 拓扑、mobility、gravity、collision 和已禁用的 converter drive，然后才执行原生 materialization、prim/view 构建、参数读回、capability 检查和 effective-report 校验。禁用 role cache 时仍会为每个 role 生成独立临时副本，因此 role 编辑不会修改缓存 raw USD 或另一个 role。

映射 IsaacSim 场景通过 IsaacLab `ContactSensor` filter 服务 `<contact data="force" reduce="netforce"/>`。数值单位为牛顿，是世界系 3 向量，表示 `geom2` 目标 body 作用在 `geom1` 来源 body 上的 filtered normal contact force；IsaacLab 已对接触点求和。`get_sensor_data()` 返回最后一个已完成物理子步，不在子步间平均；指定碰撞对无接触时返回零，reset 清理陈旧行。body-net `data="found"` 和未支持的 reduction 均快速失败。legacy 预留 body-net 槽位绝不作为碰撞对查询暴露。SDK-free 测试固定声明、wire、行映射和 reset 语义；gated real-worker 套件检查 1 kg 与 2 kg 的静态支持力。

Mapped IsaacSim 还支持作用于原生 body COM 的世界系 `body_force` 与 `body_torque` 区间 wrench。多次提交在下一次 step 前累加，作用于请求的每个子步，并在随后消费。Entity reset 清理选中实体 body，完整 reset 清理所有暂存 body 行。映射场景也接受既有宿主 `set_pre_step_control()` callback：宿主把一个公开子步对应为一次 worker STEP，并在每次 callback 前刷新共享状态。callback 的 control 与动态 wrench 每个子步从头重算，与暂存区间 wrench 叠加，并在 step 调用结束时消费。这是正确性宿主循环，不是 #152 跟踪的 device-resident controller 或性能路径；legacy model-file callback 仍不支持。

## Drake 有边界 portable profile

Drake 消费公共 portable compiler 产出的 expanded MJCF，并只在 DrakeUni 公开 model 信息与 body 查询和冻结的 `CompiledSceneLayout` 完全一致时接受布局：维度、全局 body 名称/ID、每个 root 与标量 joint qpos/qvel 映射、joint 全量消费、actuator 名称顺序及目标地址都必须精确绑定。固定 root 通过公开 body-state 查询读取；浮动 root 使用广义状态快照。被动关节贡献状态与速度，但不隐式增加控制列。

局部 entity reset 先快照完整场景状态，应用已校验 patch，再通过 DrakeUni 公开 reset 提交完整、一致的行为。未选中的实体与环境保留状态。原生 reset 失败会使 backend 进入 faulted。`restore_default_controls=True` 会在变更前被拒绝，因为 DrakeUni reset 尚未提供显式 control 恢复契约。entity 模式同样拒绝隐式单 root base/body-frame 辅助接口。

首个 Drake profile 只支持无 variant、固定或浮动 root 的 MJCF 物理实体、被动标量关节以及固定 rigid 静态实体。固定 variants 与 kinematic mirrors 会在加载 DrakeUni 或物化 portable model 前快速失败。Drake 原生验收是支持声明的必要条件；缺少 Drake native batch extension 时可选测试会跳过。DrakeUni 必须包含多实体布局排序修复（当前发布的 `drake-uni==0.1.0` 尚不包含该前置修复）。

## Adapter profiles

| Adapter | 当前 profile | 绑定与 reset 边界 |
| --- | --- | --- |
| MuJoCo | 支持含固定/浮动/kinematic 实体、镜像、被动关节和 same-layout variants 的 MJCF 源。 | 一个编译后的 `mjbatch` 场景使用冻结公共地址；局部 reset 只 scatter 受影响行并保留无关通道。 |
| MJWarp | 在 MuJoCo 组合 profile 上增加 CUDA 逐世界 variant 字段和具名编译几何。 | 单个 model/data runtime 原地上传选中值，恢复持久通道并 forward 主 Data；不声明存在选择性原生 forward。 |
| Drake | 支持无 variant、固定/浮动 root 的 MJCF 物理实体、被动关节和固定 rigid 静态实体。 | 一个 expanded portable model 在冷路径对照公开布局元数据审计；局部 reset 提交一致完整行，不支持 variants、mirrors 与 control 恢复时快速失败。 |
| IsaacGym | 支持独立 MJCF 实体、标量关节、position drive、rigid 镜像和不可变任意 assignment。 | 审计查询到的 actor/body/DoF 索引；indexed 写入在下一步前合并，后代 body 读取显式暴露新鲜度边界。 |
| IsaacSim | 支持 articulation/rigid view、标量关节、不可变同 drive K 原型 assignment、单 body 刚体、geometry 读回、映射碰撞对力传感器和暂存世界系 body wrench。 | 审计 prim/view、assignment、body/joint 映射、原生 body/geometry 属性与选中 sphere 半径；局部写保留未提及通道，提交后的失败会使 worker 进入 faulted。 |

## 验证

`tests/contract/test_worker_scene_native.py` 通过 `UNISIM_TEST_ISAACGYM_SCENE=1` 或 `UNISIM_TEST_ISAACSIM_SCENE=1` 启用真实 factory-to-worker 验收。测试覆盖两个后端的 N5/K2 非 round-robin 身份、nq/nv/nu、局部 reset 隔离与持久性、与 qpos 不同的 keyframe control、完整场景回放及版本化导入报告。各 worker 测试还覆盖独立原生质量/COM/惯量、拓扑、镜像、被动 articulation 和过滤接触力；IsaacSim gate 还会读取原生 geometry 身份、collision 状态与 PhysX material——包括同一 body 上的多个 collision prim——并从已 spawn 的 visual USD 读取选中原生 sphere 半径，同时覆盖 force、torque、消费、selected-reset wrench、逐子步 callback 状态/wrench 组合行为，以及带原生读回和缓存文件不可变校验的 raw/role USD 冷/热命中。最终 #72 测试用同一个 MJCF-only robot/object/table/mirror 操作场景，在 MuJoCo 与 gated IsaacSim 中通过公共构造、布局、属性、接触、reset、逐子步控制、wrench 与 playback 检查，并记录 runtime/GPU 证据且排除 #133 camera/RGB。IsaacSim 测试被跳过不构成原生证据。

既有 `model_file` 入口保留冷路径 importer 和源配置，随后将已初始化的原生对象交给显式实体使用的同一个场景执行器。`LegacySlotProjection` 保留历史 root/state/control 缓冲形状与名称，不包含物理循环。两个 worker 均只有一套 step、reset 和 refresh 实现。旧 D 宽动作（含被动列）与合成的 7/6 root 坐标作为显式兼容映射保留，不代表源资产声明了 free joint 或相应 actuator。Gym 历史 COM 线速度输出和世界角速度 root 槽与 canonical link/body 系坐标分别转换。既有地面/importer 策略保留在冷路径，旧 Isaac host 不新增 SDK 依赖。

Drake portable-entity 验收覆盖重复本地名称、两个浮动 root、被动关节可见性与物理响应、局部 reset 隔离、N5 batch 行与独立 N1 runtime 对比、反向声明顺序、不支持 variants/mirrors 的物化前拒绝，以及 close 或冷路径布局不匹配时的清理。Drake 测试被跳过不构成原生证据。
