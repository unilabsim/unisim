# MJWarp 实体组合与局部重置

[English](../en/m2-mjwarp.md) | [中文](m2-mjwarp.md)

## 范围与构造

本 CUDA 切片在 [#108](https://github.com/unilabsim/unisim/issues/108) 下，通过同一个既有 MJWarp model/data runtime 实现 [M2 实体契约](adr-m2-entities.md)。复用 [MuJoCo 冷路径构场 owner](m2-mujoco.md) 编译具命名空间的 MJCF 实体、校验实际公共布局并生成完整场景 variants，不为各实体创建独立 runtime。

支持声明包括固定/浮动 articulation、不增加动作的被动关节、静态刚体、kinematic 刚体及禁用碰撞的 mocap 镜像。一个物理实体可消费带显式任意 assignment 的 `same_layout` catalog。源格式、全局 option/compiler 冲突及不支持的组合遵循冷路径构场规则；MJWarp variant 校验还要求共享原生字段一致，除非已实现对应 per-world 表示。

每个 entity/root/joint/actuator 都从编译模型绑定。Fixed variant 字段通过既有 MJWarp realization 路径安装，包括质量、惯量、COM、几何边界和编译派生常量刷新。生成源保持持有到关闭，供逐环境 playback 加载完整选中场景。

当前 pinned `FixedVariantRealization` 要求每个 variant 的几何体都有唯一且非空的编译名称。Adapter 遇到该条件不满足会明确拒绝并清理生成源，不会静默按声明顺序推断几何身份或自动命名。因此具名 source geom 是当前 MJWarp profile 的前提；匿名 geom 归一化属于后续构场改进，不构成当前 runtime 支持声明。

## 默认值、身份与状态

Backend 从每环境分配的完整场景源初始化。选择具名默认 key 时，采用其中关节 qpos/qvel、控制及 activation；否则采用编译 qpos0 和零速度/控制/activation。Root 和镜像位姿来自 `EntityInitialState`，初始 root 速度为零，覆写源/keyframe root 值。对外暴露初态前，mocap 位姿、device time 及所选 key 值已上传到 main Data。

`get_scene_layout()`、`get_entity_names()` 和 `get_entity_state(name)` 使用与 CPU 构场相同的公共布局。状态快照与内部存储分离。浮动 root 的线速度参考 root link 原点；角速度由 MuJoCo 广义 body-frame 坐标转换到世界系。Kinematic 位姿来自当前 mocap 缓存。局部及完整 reset 均不改变 fixed variant identity。

缓存的构造 `ImportReport` 增加按 variant 和 assignment 环境 ID 限定范围的 `entity.initial_defaults`。Requested 值以 adapter-setting 来源记录编译场景/key 默认值。Effective 值在初始化上传和 forward 后从实际 main device Data 读回，不把组合默认值重新标记为原始源值，也不声称后续实时状态保持默认值。Reason 记录 root/keyframe 覆写规则；step/reset 后仍以当前状态查询为准。

## 局部 reset 与状态新鲜度

`reset_entities()` 首先在一致的 host 快照上调用共享 `prepare_scene_reset()` 校验器。可变提交前解析所有 patch，保留输入行顺序并使用冻结的 qpos/qvel/mocap 映射。Pose-only patch 保持世界系角速度。只清理选中状态及其关联 actuator、activation、施力和 warmstart 通道；其他实体/环境通道及 staged wrench 保留。完整 `reset(env_ids)` 恢复每环境编译/key 默认值。旧广义状态 `set_state()` 保留既有整环境 reset 语义，不静默改成局部实体事务。

固定版本的 `mujoco_warp.forward(model, data)` 没有选 world 参数。因此本实现原址上传准备值，并在**既有 main Data** 上 forward，重算全 batch 派生工作区。局部实体 reset 不使用 `reset_data()`，也不使用会索引错误 per-world variant 字段的 scratch-local world ID。这是正确性路径，不宣称 selective-forward 性能收益。

显式 reset barrier 传输保存 persistent ctrl/act/qfrc/xfrc/warmstart 通道。未修改值保持不变，forward 后恢复 warmstart，避免重计算清掉另一实体的积分历史。未选环境的 host sensor 快照不变；选中环境中未修改实体的 authored sensor 快照也保留原值，所改实体的 sensor 随 reset 刷新。由于接触耦合实体，device 派生运动学/contact 工作区可能重算；这不是新的 physics step，也不会作为未修改实体的新力测量暴露。Body 查询请求当前运动学时继续遵循既有 tracked-body 新鲜度规则。

校验失败保持状态不变。原生提交/forward 失败使 backend 进入 faulted；后续 step 及 state/body/sensor/playback 读取拒绝执行，直至重建。不承诺 GPU 回滚。

## Playback 与验证

Playback 解析所选环境的完整场景。Physics snapshot 使用既有 mocap position/quaternion 尾部，独立放置的目标可正确回放。关闭 backend 时释放持有的生成源。

在可用 GPU 上运行真实 CUDA 验收，避免与其他 vendor engine 验收同时运行：

```bash
uv sync --locked --extra mujoco --extra mjwarp
uv run --no-sync pytest -q tests/adapters/mjwarp/test_entities.py
uv run --no-sync pytest -q tests/adapters/mjwarp
```

测试覆盖 N=2/K=2 与 N=5/K=2、assignment `[1,1,0,1,0]`、原生逐环境参数读回、固定/浮动机器人和被动物体、与独立原生 MuJoCo Data 比较带 COM 偏置的 link/world 速度、乱序 reset 行、控制/activation/施力/warmstart 隔离、连续 reset 和 step、具名默认值、独立镜像位姿及完整场景 playback、失败处理、有/无镜像轨迹以及短程独立 CPU oracle rollout。CPU/CUDA 比较采用明确数值容差，不要求跨引擎轨迹逐位相同。

最终验收须记录最终 SHA、引擎/Warp 版本、GPU/driver、精确命令、容差及不支持/未验证范围。SDK 可用、契约测试或构造成功本身不构成 runtime 验证。大 batch reset 优化、更广接触语义、其余后端实现及完整 M2 审计仍是 #108 下的独立工作。
