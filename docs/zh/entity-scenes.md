# 实体场景执行

[English](../en/entity-scenes.md) | [中文](entity-scenes.md)

本文说明各 adapter 如何执行[实体决策](adr-entities.md)定义的公共实体、不可变身份和局部 reset 契约。MuJoCo 系 adapter 编译并运行完整场景；Isaac adapter 保留独立 worker 和原生执行。

旧 MuJoCo/MJWarp 编译场景也在冷路径使用相同的 `CompiledModelIndex` audit。它记录原生 body 分区、root、joint qpos/qvel 地址、mocap 地址及 actuator transmission/control 列，不重命名匿名对象，也不把 tendon/site/root transmission 假装成标量 joint actuator。旧 whole-model API 保留源语义；只有分区交叉校验通过时才暴露受限实体布局。

## 源准备与身份

共用 MJCF composer 校验源、默认值、名称和同布局 variants。独立 worker 资产写入编译器派生的显式 body 惯性参数及关节限位。单位 gear position drive 意图校验并保存为独立表后，从导出 XML 删除 actuator；原生 MJCF importer 无法安全消费 MuJoCo canonical general actuator 拼写。此 profile 明确拒绝源被动关节 damping/弹簧、activation state、不支持的 transmission 和非标量关节。编译后的逐环境 actuator 控制限位用于 step target 及初始/完整 reset control；未启用限位时不按存储的零范围夹紧。

生成文件名使用安全的内部 USD 标识，不定义公共实体身份。编译后编辑从当前 spec 序列化，避免写出旧编译结果。宿主持有生成的完整场景及独立源，直到 worker 关闭。Worker 返回的实体名称、完整 assignment 和实际实例质量均与编译意图核对。Worker audit 还确认原生拓扑、drive 和惯量采用情况。Echo 本身不是独立资产身份证据。

宿主发布独立 nq/nv/nu 和实体根布局。公共 root state 使用 link 原点位置/线速度、wxyz 四元数和世界系角速度；clone offset 由 adapter 恰好消除一次。完整广义 qvel 保留 body 系角速度。被动关节贡献状态，但不隐式增加 action 列。

## Reset 与控制

全部实体 patch 在物化或写原生状态前校验。准备好的行和显式 mask 经同一版本化 reset 命令提交。新实体入口的完整 qpos/qvel 写也归一为相同 patch 提交。完整 `reset()` 恢复选中环境的 variant 默认值和独立 keyframe control；control 不必等于关节位置。其它环境和未选实体保留状态与 targets。

IsaacGym 将 indexed root/DoF 提交累积到下一物理步，避免第二次 indexed setter 覆盖较早 reset。原生 COM 速度在公共 link 原点边界两侧转换。Root 和 joint state 立即可用；初始化或受影响 reset 后的 articulation descendant body/sensor state 在下一 step 前明确不可用。宿主拒绝相关读取，不把旧值当作当前状态。旧 model-file 路径行为不同：PhysX 不推进物理就无法刷新 link pose，因此在 INIT keyframe 或 `set_state` 之后、首个物理步之前，worker 将精确的 MJCF 前向运动学叠加到刚写入的环境上——由有效广义状态计算位置、姿态和 link 原点线速度/世界角速度——并同时清除其陈旧的 contact force 行。宿主把运动学树扫描进 INIT payload（fixed variants 下逐 variant），与公开 body/joint 布局不一致时 fail-closed。legacy reset 输入遵循 canonical 广义速度约定（世界系 link 原点线速度、body 系角速度）；历史 COM 速度投影仅保留在对外发布的 legacy 输出缓冲上。

无法恢复的原生提交失败会设置 worker fault 标记，宿主拒绝后续状态和 step 使用。提交前校验失败保留会话。共享内存槽在 attach 前校验 shape/dtype，包括零宽动作或状态布局。

## 检查与回放

Worker 必须提供版本化配置报告，宿主校验后才接受。实体 assignment 与 body mass 使用作用域记录，区分源意图与实例读回；源值不替代缺失的运行值。mapped IsaacSim 场景中，INIT 会校验原生 mass 与 COM 记录；`get_body_mass()` 再将这些不可变 worker 原生快照 scatter 到冻结的公共 body 顺序，返回独立的 `(num_envs, nbody)` 表。未被实体拥有的公共行保留编译源规范值。`get_body_ipos()` 返回形状为 `(nbody, 3)` 的独立编译源规范默认值；`get_body_ipos(env_ids=...)` 返回形状为 `(len(env_ids), nbody, 3)` 的 worker 原生物化行，并保留空选择、重复项与乱序选择。legacy model-file 路径仍快速失败。几何名称/contact mask/摩擦、接触力、属性修改、wrench 与子步控制仍不支持。回放返回选中环境的完整场景源，保留 robot、object、table 和 mirror。原生渲染仍由 worker 拥有；本切片尚未提供 mapped worker 的 physics snapshot 导出。

当前原生相机 profile 使用既有跟踪行为，拍摄环境 0 的第一个实体。录制只支持配置 `cam_distance`、`cam_elevation` 和 `cam_azimuth`。非默认的 `cam_lookat`、`cam_tracking`、`cam_tracking_env_idx`、`cam_tracking_extra_envs` 或 `cam_fov` 在访问 worker 前抛出 `NotImplementedError`，重复初始化 renderer 时也会校验。默认 `CameraCfg` 保留既有原生视图，不会选择 MuJoCo 网格相机。交互 viewer 也拒绝自定义球面偏移，其视图由原生 viewer 控制。可以返回任意选中环境的完整场景源，不代表原生相机可以选择该环境。

IsaacSim 当前支持 same-drive round-robin variant assignment 和单 body rigid view，其它组合明确拒绝。固定根的原生 root mode 和环境 view 行映射均独立 audit。共用宿主只启用已说明的 MJCF 标量关节 profile，不代表 URDF 或全部 PhysX 资产特性。

## Adapter profiles

| Adapter | 当前 profile | 绑定与 reset 边界 |
| --- | --- | --- |
| MuJoCo | 支持含固定/浮动/kinematic 实体、镜像、被动关节和 same-layout variants 的 MJCF 源。 | 一个编译后的 `mjbatch` 场景使用冻结公共地址；局部 reset 只 scatter 受影响行并保留无关通道。 |
| MJWarp | 在 MuJoCo 组合 profile 上增加 CUDA 逐世界 variant 字段和具名编译几何。 | 单个 model/data runtime 原地上传选中值，恢复持久通道并 forward 主 Data；不声明存在选择性原生 forward。 |
| IsaacGym | 支持独立 MJCF 实体、标量关节、position drive、rigid 镜像和不可变任意 assignment。 | 审计查询到的 actor/body/DoF 索引；indexed 写入在下一步前合并，后代 body 读取显式暴露新鲜度边界。 |
| IsaacSim | 支持 articulation/rigid view、标量关节、round-robin 同 drive variants 和单 body 刚体。 | 审计 prim/view 与 body/joint 映射；局部写保留未提及通道，提交后的失败会使 worker 进入 faulted。 |

## 验证

`tests/contract/test_worker_scene_native.py` 通过 `UNISIM_TEST_ISAACGYM_SCENE=1` 或 `UNISIM_TEST_ISAACSIM_SCENE=1` 启用真实 factory-to-worker 验收。测试覆盖 Gym 非 round-robin 身份、IsaacSim 已支持的 round-robin profile、nq/nv/nu、局部 reset 隔离与持久性、与 qpos 不同的 keyframe control、完整场景回放及版本化导入报告。各 worker 测试还覆盖独立原生质量/COM/惯量、拓扑、镜像和被动 articulation。

既有 `model_file` 入口保留冷路径 importer 和源配置，随后将已初始化的原生对象交给显式实体使用的同一个场景执行器。`LegacySlotProjection` 保留历史 root/state/control 缓冲形状与名称，不包含物理循环。两个 worker 均只有一套 step、reset 和 refresh 实现。旧 D 宽动作（含被动列）与合成的 7/6 root 坐标作为显式兼容映射保留，不代表源资产声明了 free joint 或相应 actuator。Gym 历史 COM 线速度输出和世界角速度 root 槽与 canonical link/body 系坐标分别转换。既有地面/importer 策略保留在冷路径，旧 Isaac host 不新增 SDK 依赖。
