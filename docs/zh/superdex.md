# SuperDex CPU 开发配置

[English](../en/superdex.md) | [中文](superdex.md)

`superdex` 适配器让 SuperDex Physics 与 Robotics 1.1.0 直接运行在 `SimBackend` 之后。该开发配置仍由 adapter owner 维护；包版本不变，本地集成不需要发布 PyPI 版本。

## 安装与所有权

使用 `superdex-uni` wheel 覆盖的 CPython 3.12 或 3.13。在 UniSim checkout 中执行以下命令同步两个可选 extra：

```sh
uv sync --python 3.12 --extra superdex --extra mujoco
```

`superdex-physics-uni==1.1.0` 与 `superdex-robotics-uni==1.1.0` 是可选依赖。它们是 SuperDex facade 的临时 unilabsim 构建，从 [unilabsim/superdex-uni](https://github.com/unilabsim/superdex-uni) tag `v1.1.0`（packaging commit `eb514cf`）发布，直到上游 SuperDex 发布等价官方 wheel。该 tag 将公共 `project_superdex` executor 契约固定在 `34a8250`；wheel metadata 记录同一 source provenance。它们安装到与上游包相同的 `superdex/` 命名空间，不能与上游包共同安装。该 extra 还提供 MuJoCo 3.11 作为冷路径 MJCF 解析器；SuperDex 执行每个物理步，原生 `.superdex_bot` 加载不使用该解析器。导入 `unisim` 或其 `SuperDexBackend` 类不会加载任一引擎。SuperDex Lab、Gymnasium 和 learner 都不是适配器依赖。

对于相邻的 UniLab checkout，保持两个版本不变，并把本地 editable 项目安装在一起，例如 `uv pip install -e './[superdex,mujoco]' -e ../UniLab`。测试 editable 覆盖时使用 `uv run --no-sync`（或 `UV_NO_SYNC=1 make check`），避免普通项目同步用索引发行包替换它们。UniLab 的本地来源测试配置使用指向精确 UniSim checkout 的 `UNILAB_LOCAL_UNISIM`。UniLab 后端指南描述其任务与注册资产配置。

已验证平台是 Linux x86_64 CPU FP32。上游也提供 Windows x86_64 和 macOS ARM wheel，但本集成尚未确立这些平台。默认 x86 构建需要 AVX2 及相关指令。上游源码暴露可选 CUDA 线性求解器，但被测试 wheel 报告为未包含 CUDA 构建；本适配器不启用 GPU 求解器。FP64 上游包要求在导入前做进程级精度选择，当前集成的数值验证面向 FP32。

每个环境拥有独立原生场景。适配器对进程全局引擎做引用计数，因此关闭一个实例不会影响其他实例。源码构建的 SuperDex `SceneBatchExecutor` 在持久 C++ worker 中批处理力写入、步进、铰接状态、link 状态、接触传感器和求解器状态。`superdex_num_workers=0` 使用进程可见的物理核心（Linux 拓扑或 macOS `sysctl`），并禁用 SDK 内部 worker。运行时初始化属于 UniSim，活跃后端不能跨进程转移。解释器关闭前调用公共 `cleanup_scene_assets()` 钩子或 `close()`；UniLab 的 `env.close()` 会调用该公共钩子。

Portable entity scene 需要 `superdex-uni` 1.1.0 提供的 `SceneBatchExecutorV2` ABI 2。其公共构造函数暴露 actor slot 的 DoF、link 与 actuator offset，以及选择性状态写入和 worker 失败语义。UniSim 在同一逐环境原生场景内为每个公开物理实体绑定恰好一个原生 actor slot。whole-MJCF 与原生 bot 路径继续使用既有 V1 单 actor executor。

## 原生调试器与串行执行

SuperDex 场景的 `DebugDraw` 对象有线程亲和性。连接原生 SuperDex 调试器时，其同步回调会从场景 step 线程收集 debug-draw 数据；因此附加调试器时在 `SceneBatchExecutor` worker 上步进场景会违反亲和性并触发原生 trap。默认 `batch` 执行模式因此快速失败：调试器客户端已连接时构造或步进后端会抛出可操作的 `RuntimeError`。

只应在串行执行模式下附加调试器；该模式不构造 executor，并在环境线程上步进每个场景：

```sh
create_backend("superdex", scene, num_envs, sim_dt, superdex_execution_mode="serial")
```

在 UniLab 中，于 Hydra 命令行传入 `env.superdex_execution_mode=serial`。`superdex_num_workers` 在串行模式无效果。该模式是调试配置，不是性能配置；训练时优先使用 `batch`。

串行模式还为 `run_playback` 的 `interactive` 渲染模式解锁原生 Polyscope viewer（`superdex.physics.viewer`）。该 viewer 与场景步进共享线程，因此交互播放要求串行模式且恰好一个环境，否则以可操作错误快速失败。`record` 与 `auto` 播放仍使用共享 MuJoCo 离线渲染器，并在两种模式下可用。UniLab 的交互 SuperDex 评估会同时注入两个设置（`serial` 加 `training.play_env_num=1`）。

## 原生固定基座机器人

预处理 SuperDex 资产保留在代码仓库之外。FR3 示例使用上游 `assets/bots/arms/fr3_v2` 目录，包括其 HDF5 碰撞与 GLB 渲染文件，并保留其 `LICENSE` 与 `NOTICE`。原生 bot 必须有硬根以及 fixed、hinge 或 slide 关节；组件、环、 tendon、传动等超出该配置的能力会快速失败。

```python
import numpy as np
from unisim import create_backend
from unisim.scene import SceneCfg

backend = create_backend(
    "superdex",
    SceneCfg("/path/to/project_superdex/assets/bots/arms/fr3_v2/fr3_v2.superdex_bot"),
    num_envs=2,
    sim_dt=0.002,
    base_name="fr3_link0",
    superdex_num_workers=0,
    superdex_effort_limits=[20, 20, 20, 20, 5, 5, 5],
)
try:
    backend.step(np.zeros((2, 7)), nsteps=5)
    state = backend.get_state()
finally:
    backend.cleanup_scene_assets()
```

原生控制向量名称和顺序遵循单 DoF 关节名称。资产中必须存在正有限力矩限制，或显式传入。示例值定义研究控制配置，不是经验证的 FR3 硬件额定值。固定基座 `get_state()` 只包含关节坐标；为固定 body 请求浮动根布局会被拒绝。命名 keyframe 必须真实存在于场景中；适配器不会为 bot 虚构 `home`。

## 已审计 MJCF 配置

冷路径导入器接受一个铰接树、一个可选自由根、hinge 与 slide 关节、标量无状态 motor 或线性 position 执行器，以及作者声明的静态 plane。既有 scene fragment 与命名 keyframe 会在步进前物化。关节与执行器顺序保持区分。质量、惯性坐标系与质心、关节坐标系与轴、armature、关节摩擦、控制和力限制都被显式映射。

Portable `entity_assets` 把该已审计 MJCF 配置扩展到无 variant、无 mirror 的物理 fixed/floating entity，并为固定静态实体使用零 DoF 原生 rigid actor。每个实体的公开 qpos/qvel、body、actuator 与接触身份都映射到冻结的原生 actor 布局。支持 entity 拥有的 geom-pair 接触传感器以及选中 reset 影响范围内的控制恢复；variant、mirror、物理 kinematic root 与 portable world-body plane 接触传感器均快速失败。

动态基本几何碰撞体在物化期间一次性三角化并烘焙为 SDF。分离的焊接几何 link 保留作者声明的 geom-pair 接触传感器身份，其质量与惯性部分求和等于原 body 的惯性属性。Mesh 碰撞、任意多关节 per body、多铰接树、equality、tendon、flex、mocap、hfield、plugin 特性，以及不支持的执行器或传感器语义会被拒绝。视觉 mesh 文件仍必须存在于源 MJCF 解析器路径中，即使该适配器无头。reset、step 或 getter 期间不发生模型解析或 SDF 烘焙。

SuperDex 接触及其隐式积分与 MuJoCo 不数值等价。基本几何 SDF 近似解析表面，求解器设置含义也不同。扭转与滚动摩擦需要显式 `superdex_allow_contact_approximation=True` 实验配置，并会警告只保留滑动 Coulomb 分量。默认实现拒绝这种语义损失。Go2 的 task owner 选择加入该配置；有限 rollout 不是运动质量或等价接触的证据。

1.1.0 wheel 缺少较新源码树的逐 pair 摩擦覆盖 API。因此导入器把作者声明的滑动摩擦 pair 分解为原生 actor 系数，使其几何平均混合复现所选 MuJoCo pair 系数。不兼容摩擦图会被拒绝；不使用私有引擎 API，也不静默修改混合规则。

## 状态、控制与传感器

公共自由根 qpos 是世界 xyz 加 wxyz 四元数，后接单 DoF 关节。公共 reset qvel 是世界 body 原点线速度加 body 坐标系角速度，后接关节速度。原生 SuperDex 自由 qpos 保存旋转向量，但它的自由旋转速度不是该向量的普通导数。在原生参考变换为单位变换时，原生自由 qvel 使用世界原点线速度和世界角速度。适配器在状态屏障处旋转角速度分量，并在非平凡姿态对照作者声明的 MuJoCo 运动学验证 body 原点与质心速度。

pre-step 控制回调每个物理子步运行一次。Motor 与 position 控制尊重作者声明的顺序、增益、gear 和限制。待处理 body 力会累积为广义力并与控制一起提交；一次原生外力写入不能抹除单独的控制贡献。

命名关节位置与速度、frame pose、轴与速度、gyro 以及 velocimeter 信号从原生状态重建到 NumPy 缓存。支持的 plane 与 geom `contact data="found" num="1"` 信号使用原生接触点和实际 actor pair，而不是非零力代理。接触表示最后一次完成的物理解算。reset 会清除已解算接触状态；传送后 `step(0)` 不会重建接触流形，因此第一个正物理步提供新的接触结果。不要把 reset 时的接触标志当作几何重叠测试。

作者声明的加速度计会被识别但不可用：请求或绑定它会抛出 `NotImplementedError`，因为公共运行时不提供瞬时点加速度。未使用的加速度计不会阻止加载其他方面受支持的资产，也不会把零值或有限差分替代品呈现为作者传感器。原生 bot 传感器组件、相机、任意力与触觉传感器、site Jacobian 都不属于该配置。

完整 reset 恢复私有初始动态快照，写入选中的 qpos 与 qvel，清除控制和外力，并刷新运动学缓存。其他行保持不变。Portable 局部 entity reset 保留无关环境、实体与控制；它只清空目标为本次 reset root/joint 字段的控制，`restore_default_controls=True` 会把恰好这些列恢复为零构造默认值，或在选中命名默认 keyframe 时恢复该合并 keyframe 经限幅的控制值。快照字节不会作为可移植 checkpoint 暴露。模型域随机化、渲染与视频、ROM、soft 与 tactile 状态、GPU 批处理物理不受支持，调用者不得宣传这些能力。存在可视 MJCF 模型时，播放使用共享离线 MuJoCo 渲染器。

## 验证

`scripts/benchmarks/superdex_scene_step.py` 是维护者专用的原生物理屏障测量工具，用于 direct scene stepping 与 batch executor 对比。它排除模型加载、动作、观测、奖励、reset、collector 和 learner，因此不是 RL 吞吐 benchmark。

```sh
uv run --no-sync pytest -q tests/adapters/superdex/test_contract.py tests/adapters/superdex/test_backend.py tests/adapters/superdex/test_materialization.py tests/adapters/superdex/test_portable_scene.py
UV_NO_SYNC=1 make check
uv lock --check
make package
```

将 `SUPERDEX_ASSETS_PATH` 指向上游 `assets` 目录可包含外部原生 FR3 fixture。其他数值测试使用小型手写模型，并需要可选 Python 3.12 运行时；contract 与 import 测试也可以在没有它的情况下运行。UniLab 拥有 task rollout、训练 checkpoint 与 sim2sim 策略 I/O 验证，这些结果记录在 roadmap 的 integration 子任务中。
