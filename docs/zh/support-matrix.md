# 适配器支持矩阵

[English](../en/support-matrix.md) | [中文](support-matrix.md)

| 后端 | 公共类 | 安装与运行时边界 | 状态 |
| --- | --- | --- | --- |
| MuJoCo | `unisim.MuJoCoBackend` | `uv sync --extra mujoco`（mjbatch 运行时） | available |
| Motrix | `unisim.MotrixBackend` | `uv sync --extra motrix` | available |
| Drake | `unisim.DrakeBackend` | `uv sync --extra drake`（`drake-uni`）及其原生批处理扩展 | available |
| MJWarp | `unisim.MJWarpBackend` | `uv sync --extra mjwarp`，CUDA | available |
| Genesis | `unisim.GenesisBackend` | `uv sync --extra genesis` | available |
| Newton | `unisim.NewtonBackend` | `uv sync --extra newton`，Newton 1.5.1 与 MuJoCo-Warp 3.11.0 | available（CUDA） |
| SuperDex | `unisim.SuperDexBackend` | `uv sync --extra superdex`，CPython 3.12 或 3.13，SuperDex 1.0.0 | 实验性 CPU；见[配置说明](superdex.md) |
| IsaacGym | `unisim.IsaacGymBackend` | `uv sync --extra isaacgym`（空 extra）加专用 Python 3.8 worker | available |
| IsaacSim | `unisim.IsaacSimBackend` | `uv sync --extra isaacsim`（空 extra）加专用 IsaacSim 或 IsaacLab worker | available |

基础 wheel 不导入以上任何 SDK。构造执行冷路径运行时发现，并在运行不可用时抛出适配器专属、可操作的错误。本矩阵是适配器与 API 支持声明，不是每台主机都具备每个厂商 SDK 或 GPU 能力的声明。

MuJoCo 适配器的原生执行器是 [mjbatch](https://github.com/unilabsim/mjbatch_uni)，即 `kevinzakka/mjbatch` 的维护 fork；它为 Linux x86_64、aarch64 和 macOS（CPython 3.10 到 3.14t）提供预构建 wheel，并精确固定 `mujoco==3.11.0`。原生执行器不支持 Windows 和 musllinux，因此 Windows CI 作业只运行核心与导入边界子集。切换到当前执行器前后的数值结果不保证相同；漂移由已记录基线表征，而不是位级精确门禁。适配器支持构造时 `FixedVariantPlan` 目录及 `same_layout` 与 `uniform_public_layout` 保证。Same-layout 变体和可选命名 mesh-geom 槽位通过 `VariantPack` 合并到一个规范 mjbatch 执行器；异构公共拓扑快速失败。重置模型字段写入与逐世界编译器默认值使用 mjbatch `expand` 与 `set_const`，播放暴露逐环境独立编译的视觉 oracle。`chunk_size` 与 `adaptive_chunk_size` 是已弃用的 warn-and-ignore 参数；chunk 调度器已移除，mjbatch 的工作窃取线程池是调优机制。

MuJoCo 相关 extra 共享同一条版本线（MuJoCo 3.11、MuJoCo-Warp 3.11 和 warp-lang 1.16.0），可以联合安装。`mjwarp` 用 `mujoco-warp~=3.11.0` 跟踪该版本线，而 `newton` 保留与上游精确耦合的固定版本（`newton==1.5.1`、`mujoco-warp==3.11.0`、`mujoco==3.11.0`、`warp-lang==1.16.0`）。安装后运行 `uv run scripts/diagnostics/check_newton_runtime.py` 执行仅元数据探测；需要显式导入原生运行时时添加 `--import`。Newton 冷路径校准会采样求解器计数，并在 `nconmax` 或 `njmax` 过小时抛出显式容量错误；它绝不接受静默约束截断。

Newton 支持 CUDA graph 显式开启：`NewtonBackend(..., use_cuda_graph=True)` 或 `create_backend(..., newton_use_cuda_graph=True)`。只有冷路径容量校准重建最终固定地址 state 之后才会捕获 graph，并按 Newton 输入/输出 state 的奇偶交替各捕获一张。捕获要求 CUDA 设备、12.4 及以上驱动和已启用的 CUDA mempool；否则 Newton 会发出带原因的 `RuntimeWarning` 并保持 eager 执行。捕获失败同样回退 eager。state reset 与已注册的 pre-step control callback 保持 eager；无 callback 的物理步按当前 state 奇偶选择并 replay graph。

Newton 播放在只安装单个 `newton` extra 时通过 `ViewerGL`（`pyglet>=2.1.6,<3` 与 `imgui-bundle>=1.92.0`）原生渲染：`record` 离屏渲染，`interactive` 打开窗口 viewer，`auto` 根据显示可用性选择。运行时不完整时，`record` 回退到离线 MuJoCo snapshot 管线，`interactive` 以可操作错误快速失败。无头离屏 GL 需要 EGL（`PYOPENGL_PLATFORM=egl`），或在 Wayland 下使用 GLX。

## 语义清单

下表由 `src/unisim/support.py` 中的 `get_adapter_capabilities()` 生成；运行 `uv run scripts/diagnostics/check_support.py --check-docs` 校验，或用 `--write-docs` 同步生成两种语言。这些是源码审查声明，不是真实运行验证。`exact` 仅针对所述子集；`approximate` 要求逐项授权，`unsupported` 拒绝对应请求，`unknown` 不承诺支持。`*` 表示必须满足声明中的配置条件；原因、条件和固定版本源码证据可从公共报告查询。当前只声明 default profile；未知 profile 保持未知。

<!-- semantic-inventory:start -->
| Feature | mujoco | motrix | drake | mjwarp | newton | superdex | genesis | isaacgym | isaacsim |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `asset.mjcf` | exact | exact | exact | exact | exact | exact | exact | exact | exact |
| `asset.urdf` | unknown | unknown | unknown | unknown | unknown | unknown | unknown | unsupported | unsupported |
| `entity.single_articulation` | exact | exact | exact | exact | exact | exact | exact | exact | exact |
| `entity.multiple` | exact* | unknown | unknown | exact* | unknown | unknown | unknown | exact* | exact* |
| `root.free` | exact | exact | exact | exact | exact | exact | exact | exact | exact |
| `root.fixed` | exact | exact | exact | exact | unsupported | exact | unknown | unknown | unknown |
| `joint.hinge` | exact | exact | exact | exact | exact | exact | exact | exact | exact |
| `joint.slide` | exact | exact | exact | exact | exact | exact | exact | exact | exact |
| `joint.ball` | exact | unknown | unknown | exact | unknown | unsupported | unknown | unknown | unknown |
| `actuator.motor` | exact | unknown | unknown | exact | exact | exact | unsupported | unsupported | unsupported |
| `actuator.position` | exact | exact | unknown | exact | unknown | unknown | exact | exact | exact |
| `collision.rigid` | exact | exact | exact | exact | exact | approximate* | exact | exact | exact |
| `collision.self` | exact | unknown | unknown | exact | unknown | unknown | unknown | unsupported | unsupported |
| `contact.query` | exact | unknown | unknown | exact | exact* | approximate* | approximate* | approximate* | approximate* |
| `terrain.heightfield` | exact | exact | unknown | exact | unknown | unsupported | unknown | unknown | unknown |
| `sensor.imu` | exact | unknown | unknown | exact | approximate | approximate | approximate | unsupported | unsupported |
| `sensor.gyro` | unknown | unknown | unknown | unknown | unknown | unknown | unknown | approximate | approximate |
| `reset.state` | exact | exact | exact | exact | exact | exact | exact | exact | exact |
| `dr.interval.body_force` | unknown | unknown | unknown | unknown | unknown | unknown | unknown | unknown | unknown |
| `state.final_refresh` | exact* | unknown | unknown | exact | unknown | unknown | unknown | unknown | unknown |
| `state.callback_refresh` | exact* | unknown | unknown | exact | unknown | unknown | unknown | unknown | unknown |
| `variant.same_layout` | unknown | unknown | unknown | unknown | unknown | unknown | unknown | unknown | unknown |
<!-- semantic-inventory:end -->

DR、播放、body wrench 和 fixed-variant 能力仍由既有实例 API 提供权威信息。静态清单有意将依赖这些来源的项目保留为 unknown；`backend.get_capabilities()` 聚合实例权威声明。多个逻辑实体分区不代表任意多 articulation 组合。URDF 调研和未合并分支不构成当前支持。IsaacSim legacy 路径预留的零接触缓冲区既不代表有效接触查询，也不代表没有物理接触；只有映射场景中的 `contact data="force" reduce="netforce"` 声明使用专用 PhysX 碰撞对力槽位。Isaac worker 的传感器支持 gyro 重建，但拒绝 accelerometer。

[能力设计决策](adr-capabilities.md) 定义证据匹配和快照生命周期。上方安装表中的 `available` 始终不能用于判断任务兼容性。
