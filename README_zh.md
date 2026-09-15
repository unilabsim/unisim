# UniSim

[English](README.md) | [中文](README_zh.md)

UniSim 为机器人学习与仿真提供后端中立的物理契约和可选引擎适配器。PyPI 发行包是 `unisim-core`，Python 导入命名空间是 `unisim`。

一个 `SimBackend` 契约覆盖状态访问、控制、重置和域随机化边界，使同一份任务代码可以使用 MuJoCo、Motrix、Drake、MJWarp、Genesis、Newton、SuperDex、IsaacGym 或 IsaacSim，而不需要编写引擎分支。基础安装只依赖 NumPy；每个引擎 SDK 都是懒加载的可选 extra，导入 `unisim` 不会导入任何引擎。

## 与 UniLab 的关系

UniSim 是从 UniLab 抽取的后端中立物理层。UniLab 保留 Hydra 配置、task/env/manager 生命周期、机器人资产、RL 训练、checkpoint 和 sim2sim 策略 I/O；UniSim 拥有物理契约、适配器生命周期与状态转换、可选运行时诊断，以及共享的子进程 IPC 层。每个后端只有一个由本仓库拥有的生产实现。UniLab 只组装任务拥有的场景与配置输入并消费公共契约，UniSim 永不导入 UniLab。

## 安装

```bash
pip install unisim-core                         # 基础契约、工厂和 fake backend
pip install "unisim-core[<adapter-extra>]"      # 按需加入一个可选引擎运行时
```

可用 extra 为 `mujoco`、`motrix`、`drake`、`mjwarp`、`genesis`、`newton`、`superdex`、`isaacgym` 和 `isaacsim`。Isaac extra 是空拼写，因为相应厂商 SDK 不能再分发；它们的适配器会在构造时发现专用 worker 安装。所有适配器的安装、平台、CUDA 和执行器边界记录在[支持矩阵](docs/zh/support-matrix.md)。实验性 SuperDex CPU 配置有单独的[开发指南](docs/zh/superdex.md)。

## 快速开始

公共边界刻意保持懒加载，可以在任意位置安全导入：

```python
from unisim import SimBackend, create_backend
```

通过工厂和包中立的 `SceneCfg` 构造适配器：

```python
backend = create_backend("mujoco", scene=scene_cfg, num_envs=64, sim_dt=0.01)
backend.materialize()      # 冷路径：解析模型并构建引擎对象
backend.reset()
state = backend.get_state()
backend.step(ctrl)         # 热路径：校验数组和缓存句柄
```

每个适配器在可选运行时缺失时都会给出可操作的后端专属诊断并快速失败；后端不会被静默降级到另一个引擎。引擎原生 model/data 对象不会逃出适配器，状态和控制都通过校验过的 NumPy 数组传递。

确定性 `FakeBackend` 和 `assert_backend_conformance` 辅助函数让消费者无需安装引擎即可测试任务代码。`BenchmarkCase` 和 `BenchmarkResult` 是为未来 benchmark 包保留的 schema 扩展点；本仓库不实现负载运行器。

外部 worker 根目录可通过 `UNISIM_ISAACGYM_HOME`、`UNISIM_ISAACGYM_PYTHON`、`UNISIM_ISAACSIM_HOME` 和 `UNISIM_ISAACSIM_PYTHON` 配置。旧 `UNILAB_*` 拼写仍作为迁移回退保留。

## 文档

- [架构](docs/zh/architecture.md) — 所有权边界与热/冷路径规则
- [适配器支持矩阵](docs/zh/support-matrix.md) — 安装、运行时、平台和播放要求
- [UniLab 迁移](docs/zh/migration.md) — 从历史上的 UniLab 后端层迁移
- [Mocap 与重置随机化](docs/zh/mocap-reset-contract.md) — 选中环境的 mocap 姿态与 MJWarp 重置字段
- [Benchmark API 预留](docs/zh/benchmark-api.md) — 预留的 benchmark 结果 schema
- [发布手册](docs/zh/release.md) — TestPyPI 检查与自动化生产发布
- [SuperDex CPU 配置](docs/zh/superdex.md) — 实验性适配器与原生资产细节

## 开发

```bash
make sync       # 锁定环境并加入 MuJoCo 测试 extra
make check      # Ruff 与 pytest
make package    # 构建用于检查的本地发行包
```

每个适配器发布前都必须记录支持的 Python、平台和运行时矩阵，并通过一致性辅助检查。仓库布局以及 `scripts/` 与 `tests/` 子目录的用途见 [AGENTS.md](AGENTS.md)、[scripts/README.md](scripts/README.md) 和 [tests/README.md](tests/README.md)。

## 引用

如果 UniSim 对您的研究有帮助，请引用 UniLab 论文：

```bibtex
@article{jia2026unilab,
  title   = {UniLab: A Heterogeneous Architecture for Robot RL Beyond GPU-Dominant Paradigms},
  author  = {Yufei Jia and Zhanxiang Cao and Mingrui Yu and Heng Zhang and Shenyu Chen and Dixuan Jiang and Meng Li and Xiaofan Li and Yiyang Liu and Junzhe Wu and Zheng Li and XiLin Fang and Tingyu Cui and Shengcheng Fu and Haoyang Li and Anqi Wang and Zifan Wang and Dongjie Zhu and Chenyu Cao and Zhenbiao Huang and Ziang Zheng and Jie Lu and Xin Ma and Zhengyang Wei and Xiang Zhao and Tianyue Zhan and Ye He and Yuxiang Chen and Yizhou Jiang and Yue Li and Haizhou Ge and Yuhang Dong and Fan Jia and Ziheng Zhang and Meng Zhang and Xiwa Deng and Zhixing Chen and Hanyang Shao and Chenxin Dong and Yixuan Li and Yizhi Chen and Bokui Chen and Kaifeng Zhang and Hanqing Cui and Yusen Qin and Ruqi Huang and Lei Han and Tiancai Wang and Xiang Li and Yue Gao and Guyue Zhou},
  journal = {arXiv preprint arXiv:2605.30313},
  year    = {2026},
  url     = {https://arxiv.org/abs/2605.30313}
}
```

### 物理后端

通过 UniSim 使用具体后端时，请同时引用对应引擎。`mujoco` 适配器运行在 mjbatch 运行时之上，`drake` 适配器运行在 DrakeUni 运行时之上，因此请与原版引擎一并引用：

```bibtex
% MuJoCo
@inproceedings{todorov2012mujoco,
  title     = {MuJoCo: A Physics Engine for Model-Based Control},
  author    = {Todorov, Emanuel and Erez, Tom and Tassa, Yuval},
  booktitle = {2012 IEEE/RSJ International Conference on Intelligent Robots and Systems},
  pages     = {5026--5033},
  year      = {2012},
  doi       = {10.1109/IROS.2012.6386109}
}

% mjbatch (runtime of the `mujoco` adapter; UniLab-maintained fork
% of kevinzakka/mjbatch)
@software{mjbatch,
  title  = {mjbatch: Batched MuJoCo Simulation},
  author = {Kevin Zakka},
  year   = {2026},
  url    = {https://github.com/kevinzakka/mjbatch},
  note   = {The `mujoco` adapter pins the UniLab-maintained fork unilabsim/mjbatch_uni}
}

% MotrixSim
@software{motrixsim2026,
  title  = {MotrixSim: A Physics Simulation Engine for Robotics and Embodied AI},
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

% DrakeUni (runtime of the `drake` adapter)
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

% Newton
@software{newton2025,
  title  = {Newton: GPU-accelerated physics simulation for robotics and simulation research},
  author = {{Newton Contributors}},
  year   = {2025},
  url    = {https://github.com/newton-physics/newton}
}

% Genesis
@misc{genesis,
  title  = {Genesis: A Universal and Generative Physics Engine for Robotics and Beyond},
  author = {Genesis Authors},
  month  = {December},
  year   = {2024},
  url    = {https://github.com/Genesis-Embodied-AI/Genesis}
}

% Isaac Gym
@inproceedings{makoviychuk2021isaacgym,
  title     = {Isaac Gym: High Performance GPU-Based Physics Simulation for Robot Learning},
  author    = {Makoviychuk, Viktor and Wawrzyniak, Lukasz and Guo, Yunrong and Lu, Michelle and Storey, Kier and Macklin, Miles and Hoeller, David and Rudin, Nikita and Allshire, Arthur and Handa, Ankur and State, Gavriel},
  booktitle = {Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks},
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

Apache-2.0；见 [LICENSE](LICENSE)。
