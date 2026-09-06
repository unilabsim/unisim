# UniSim

[English](README.md) | [中文](README_zh.md)

UniSim 提供后端中立的物理仿真契约（contract）和可选引擎适配器，面向机器人
学习与仿真。PyPI 分发名为 `unisim-core`,Python 导入命名空间为 `unisim`。

统一的 `SimBackend` 契约覆盖状态访问、控制、重置与域随机化边界，同一份任务
代码无需引擎专属分支即可运行在 MuJoCo、Motrix、Drake、MJWarp、Genesis、Newton、
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

可用 extra:`mujoco`、`motrix`、`drake`、`mjwarp`、`genesis`、`newton`、`isaacgym`、
`isaacsim`。Isaac 两个 extra 是空声明,因为这些厂商 SDK 不可再分发;对应
适配器在构造时发现独立的 worker 安装。完整的适配器支持矩阵见
[`docs/support-matrix.md`](docs/support-matrix.md)。

`mujoco`、`mjwarp`、`newton` 三个 extra 共享 MuJoCo 3.11 / MuJoCo-Warp 3.11 /
warp-lang 1.16.0 版本线，可以安装在同一环境中。`newton` 保留精确版本固定
（`newton==1.5.1` 及其耦合运行时）,`mjwarp` 则通过 `mujoco-warp~=3.11.0`
跟随 3.11 版本线。安装后可运行
`uv run scripts/check_newton_runtime.py` 做元数据探针，必要时追加 `--import`
显式导入原生运行时。

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

## 引用

如果 UniSim 对您的研究有帮助,请引用 UniLab 论文:

```bibtex
@article{jia2026unilab,
  title   = {UniLab: A Heterogeneous Architecture for Robot RL Beyond
             GPU-Dominant Paradigms},
  author  = {Yufei Jia and Zhanxiang Cao and Mingrui Yu and Heng Zhang and
             Shenyu Chen and Dixuan Jiang and Meng Li and Xiaofan Li and
             Yiyang Liu and Junzhe Wu and Zheng Li and XiLin Fang and
             Tingyu Cui and Shengcheng Fu and Haoyang Li and Anqi Wang and
             Zifan Wang and Dongjie Zhu and Chenyu Cao and Zhenbiao Huang and
             Ziang Zheng and Jie Lu and Xin Ma and Zhengyang Wei and
             Xiang Zhao and Tianyue Zhan and Ye He and Yuxiang Chen and
             Yizhou Jiang and Yue Li and Haizhou Ge and Yuhang Dong and
             Fan Jia and Ziheng Zhang and Meng Zhang and Xiwa Deng and
             Zhixing Chen and Hanyang Shao and Chenxin Dong and Yixuan Li and
             Yizhi Chen and Bokui Chen and Kaifeng Zhang and Hanqing Cui and
             Yusen Qin and Ruqi Huang and Lei Han and Tiancai Wang and
             Xiang Li and Yue Gao and Guyue Zhou},
  journal = {arXiv preprint arXiv:2605.30313},
  year    = {2026},
  url     = {https://arxiv.org/abs/2605.30313}
}
```

### 物理后端

通过 UniSim 使用某个具体后端时,请同时引用对应的引擎。`mujoco` 与
`drake` 适配器分别构建在 MuJoCoUni 和 DrakeUni 运行时之上,因此请与原版
引擎一并引用:

```bibtex
% MuJoCo
@inproceedings{todorov2012mujoco,
  title     = {MuJoCo: A Physics Engine for Model-Based Control},
  author    = {Todorov, Emanuel and Erez, Tom and Tassa, Yuval},
  booktitle = {2012 IEEE/RSJ International Conference on Intelligent Robots
               and Systems},
  pages     = {5026--5033},
  year      = {2012},
  doi       = {10.1109/IROS.2012.6386109}
}

% MuJoCoUni(`mujoco` 适配器的运行时)
@article{jia2026mujocouni,
  title   = {MuJoCoUni: Persistent Batched Runtime Primitives for MuJoCo},
  author  = {Jia, Yufei and Wu, Junzhe},
  journal = {arXiv preprint arXiv:2605.24922},
  year    = {2026}
}

% MotrixSim
@software{motrixsim2026,
  title  = {MotrixSim: A Physics Simulation Engine for Robotics and
            Embodied AI},
  author = {{Motphys Team}},
  year   = {2026},
  url    = {https://motrixsim.readthedocs.io/},
  note   = {Python binary package}
}

% Drake
@misc{tedrake2019drake,
  title  = {Drake: Model-Based Design and Verification for Robotics},
  author = {Russ Tedrake and the Drake Development Team},
  year   = {2019},
  url    = {https://drake.mit.edu}
}

% DrakeUni(`drake` 适配器的运行时)
@software{drakeuni,
  title  = {DrakeUni: Experimental Drake Batch Simulation Runtime for UniLab},
  author = {{UniLab Team}},
  year   = {2026},
  url    = {https://pypi.org/project/drake-uni/},
  note   = {Python binary package}
}

% MJWarp
@software{mujoco_warp,
  title  = {MuJoCo Warp: A GPU-Accelerated MuJoCo Backend},
  author = {{Google DeepMind}},
  year   = {2025},
  url    = {https://github.com/google-deepmind/mujoco_warp}
}

% Genesis
@misc{genesis,
  title  = {Genesis: A Universal and Generative Physics Engine for Robotics
            and Beyond},
  author = {Genesis Authors},
  month  = {December},
  year   = {2024},
  url    = {https://github.com/Genesis-Embodied-AI/Genesis}
}

% Isaac Gym
@inproceedings{makoviychuk2021isaacgym,
  title     = {Isaac Gym: High Performance GPU-Based Physics Simulation for
               Robot Learning},
  author    = {Makoviychuk, Viktor and Wawrzyniak, Lukasz and Guo, Yunrong and
               Lu, Michelle and Storey, Kier and Macklin, Miles and
               Hoeller, David and Rudin, Nikita and Allshire, Arthur and
               Handa, Ankur and State, Gavriel},
  booktitle = {Proceedings of the Neural Information Processing Systems Track
               on Datasets and Benchmarks},
  year      = {2021}
}

% Isaac Sim
@software{nvidia2022isaacsim,
  title  = {NVIDIA Isaac Sim},
  author = {{NVIDIA}},
  year   = {2022},
  url    = {https://developer.nvidia.com/isaac/sim}
}
```

## 许可证

Apache-2.0,见 [LICENSE](LICENSE)。
