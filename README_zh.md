# UniSim

[English](README.md) | [中文](README_zh.md)

UniSim 提供后端中立的物理仿真契约（contract）和可选引擎适配器，面向机器人
学习与仿真。PyPI 分发名为 `unisim-core`,Python 导入命名空间为 `unisim`。

统一的 `SimBackend` 契约覆盖状态访问、控制、重置与域随机化边界，同一份任务
代码无需引擎专属分支即可运行在 MuJoCo、Motrix、Drake、MJWarp、Genesis、
IsaacGym 或 IsaacSim 之上。基础安装仅依赖 NumPy;所有引擎 SDK 都是懒加载的
可选 extra,`import unisim` 不会导入任何引擎。

## 与 UniLab 的关系

UniSim 是从 UniLab 中抽取出来的、后端中立的物理层，供 UniLab 使用。UniLab
保留 Hydra 配置、任务/环境/管理器生命周期、机器人资产、强化学习训练、
checkpoint 以及 sim2sim 策略 I/O;UniSim 负责物理契约、适配器生命周期与状态
转换、可选运行时诊断，以及共享的子进程 IPC 层。每个后端只有一份生产实现，
由本仓库持有——UniLab 只组装任务侧的场景与配置输入，并消费公共契约。
UniSim 不会导入 UniLab。

## 安装

```bash
pip install unisim-core                # 基础包:契约、工厂、fake 后端
pip install "unisim-core[mujoco]"      # 需要时再加装引擎 extra
```

可用 extra:`mujoco`、`motrix`、`drake`、`mjwarp`、`genesis`、`isaacgym`、
`isaacsim`。Isaac 两个 extra 是空声明,因为这些厂商 SDK 不可再分发;对应
适配器在构造时发现独立的 worker 安装。完整的适配器支持矩阵见
[`docs/support-matrix.md`](docs/support-matrix.md)。

## 快速上手

公共导入边界刻意保持惰性,可以在任何环境安全导入:

```python
from unisim import SimBackend, create_backend
```

通过工厂和包中立的 `SceneCfg` 构造后端:

```python
backend = create_backend("mujoco", scene=scene_cfg, num_envs=64, sim_dt=0.01)
backend.materialize()      # 冷路径:解析 XML,构建引擎对象
backend.reset()
state = backend.get_state()
backend.step(ctrl)         # 热路径:已校验数组与缓存句柄
```

当可选运行时缺失时,每个适配器都会给出后端专属、可操作的错误诊断——任何
后端都不会被静默降级为其他引擎。引擎原生的 model/data 对象不会逃出适配器;
状态与控制通过校验过的 NumPy 数组传递。

确定性的 `FakeBackend` 和 `assert_backend_conformance` 辅助函数让消费者无需
安装任何引擎即可测试任务代码。`BenchmarkCase` 和 `BenchmarkResult` 是为未来
benchmark 包保留的 schema 扩展点,本仓库不实现负载运行器。

外部 worker 根目录可通过 `UNISIM_ISAACGYM_HOME`、`UNISIM_ISAACGYM_PYTHON`、
`UNISIM_ISAACSIM_HOME`、`UNISIM_ISAACSIM_PYTHON` 配置。包同时兼容旧的
`UNILAB_*` 拼写作为迁移回退。

## 文档

- [`docs/architecture.md`](docs/architecture.md) — UniSim 与 UniLab 的
  所有权边界
- [`docs/support-matrix.md`](docs/support-matrix.md) — 适配器安装与运行时
  要求
- [`docs/migration.md`](docs/migration.md) — 从历史上的 UniLab 后端层迁移
- [`docs/benchmark-api.md`](docs/benchmark-api.md) — 保留的 benchmark schema
- [`docs/release.md`](docs/release.md) — TestPyPI 与自动化生产发布流程

## 开发

```bash
make sync       # 锁定环境,附带 MuJoCo 测试 extra
make check      # Ruff + pytest
make package    # 本地构建 sdist 与 wheel 检查
```

每个适配器在发布前都必须记录其支持的 Python/平台/运行时矩阵,并通过一致性
(conformance)检查。

## 许可证

Apache-2.0,见 [LICENSE](LICENSE)。
