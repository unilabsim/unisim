# UniSim

[English](README.md) | [中文](README_zh.md)

UniSim provides backend-neutral physics contracts and optional engine adapters for robot learning and simulation. The PyPI distribution is `unisim-core`; the Python import namespace is `unisim`.

A single `SimBackend` contract covers state access, control, reset, and domain-randomization boundaries, so the same task code can use MuJoCo, Motrix, Drake, MJWarp, Genesis, Newton, SuperDex, IsaacGym, or IsaacSim without engine-specific branches. The base install depends only on NumPy; every engine SDK is an optional extra loaded lazily, and importing `unisim` never imports an engine.

## Relationship to UniLab

UniSim is the extracted, backend-neutral physics layer used by UniLab. UniLab retains Hydra configuration, task/env/manager lifecycle, robot assets, RL training, checkpoints, and sim2sim policy I/O; UniSim owns the physics contract, adapter lifecycle and state translation, optional-runtime diagnostics, and the shared subprocess IPC layer. There is exactly one production implementation of each backend, owned by this repository. UniLab only assembles task-owned scene and configuration inputs and consumes the public contract, and UniSim never imports UniLab.

## Installation

```bash
pip install unisim-core                         # base contract, factory, and fake backend
pip install "unisim-core[<adapter-extra>]"      # add one optional engine runtime
```

Available extras are `mujoco`, `motrix`, `drake`, `mjwarp`, `genesis`, `newton`, `superdex`, `isaacgym`, and `isaacsim`. The Isaac extras are empty spellings because their vendor SDKs are not redistributable; their adapters discover dedicated worker installations at construction time. Installation, platform, CUDA, and executor boundaries for every adapter are recorded in the [support matrix](docs/en/support-matrix.md). The experimental SuperDex CPU profile has a dedicated [development guide](docs/en/superdex.md).

## Quick start

The public boundary is deliberately lazy and safe to import anywhere:

```python
from unisim import SimBackend, create_backend
```

Construct an adapter through the factory with a package-neutral `SceneCfg`:

```python
backend = create_backend("mujoco", scene=scene_cfg, num_envs=64, sim_dt=0.01)
backend.materialize()      # cold path: parse the model and build engine objects
backend.reset()
state = backend.get_state()
backend.step(ctrl)         # hot path: validated arrays and cached handles
```

Each adapter fails closed with an actionable, backend-specific diagnostic when its optional runtime is missing; no backend is silently downgraded to another engine. Engine-native model and data objects never escape the adapter, and state and control flow through validated NumPy arrays.

A deterministic `FakeBackend` and the `assert_backend_conformance` helper let consumers test task code without an engine installed. `BenchmarkCase` and `BenchmarkResult` are reserved schema extension points for a future benchmark package; this repository does not implement a workload runner.

External worker roots can be configured with `UNISIM_ISAACGYM_HOME`, `UNISIM_ISAACGYM_PYTHON`, `UNISIM_ISAACSIM_HOME`, and `UNISIM_ISAACSIM_PYTHON`. The former `UNILAB_*` spellings remain available as a migration fallback.

## Documentation

- [Architecture](docs/en/architecture.md) — ownership boundaries and hot/cold-path rules
- [Adapter support matrix](docs/en/support-matrix.md) — installation, runtime, platform, and playback requirements
- [UniLab migration](docs/en/migration.md) — moving from the historical UniLab backend layer
- [Mocap and reset randomization](docs/en/mocap-reset-contract.md) — selected-world mocap poses and MJWarp reset fields
- [Benchmark API reservation](docs/en/benchmark-api.md) — reserved benchmark result schemas
- [Release runbook](docs/en/release.md) — TestPyPI checks and automated production publishing
- [SuperDex CPU profile](docs/en/superdex.md) — experimental adapter and native-asset details

## Development

```bash
make sync       # locked environment plus the MuJoCo test extra
make check      # Ruff plus pytest
make package    # build the local distribution for inspection
```

Every adapter must document its supported Python, platform, and runtime matrix and pass the conformance helper before it is published. Repository layout and the purpose of the `scripts/` and `tests/` subdirectories are described in [AGENTS.md](AGENTS.md), [scripts/README.md](scripts/README.md), and [tests/README.md](tests/README.md).

## Citation

If UniSim contributes to your research, please cite the UniLab paper:

```bibtex
@article{jia2026unilab,
  title   = {UniLab: A Heterogeneous Architecture for Robot RL Beyond GPU-Dominant Paradigms},
  author  = {Yufei Jia and Zhanxiang Cao and Mingrui Yu and Heng Zhang and Shenyu Chen and Dixuan Jiang and Meng Li and Xiaofan Li and Yiyang Liu and Junzhe Wu and Zheng Li and XiLin Fang and Tingyu Cui and Shengcheng Fu and Haoyang Li and Anqi Wang and Zifan Wang and Dongjie Zhu and Chenyu Cao and Zhenbiao Huang and Ziang Zheng and Jie Lu and Xin Ma and Zhengyang Wei and Xiang Zhao and Tianyue Zhan and Ye He and Yuxiang Chen and Yizhou Jiang and Yue Li and Haizhou Ge and Yuhang Dong and Fan Jia and Ziheng Zhang and Meng Zhang and Xiwa Deng and Zhixing Chen and Hanyang Shao and Chenxin Dong and Yixuan Li and Yizhi Chen and Bokui Chen and Kaifeng Zhang and Hanqing Cui and Yusen Qin and Ruqi Huang and Lei Han and Tiancai Wang and Xiang Li and Yue Gao and Guyue Zhou},
  journal = {arXiv preprint arXiv:2605.30313},
  year    = {2026},
  url     = {https://arxiv.org/abs/2605.30313}
}
```

### Physics backends

When you use a specific backend through UniSim, please also cite the corresponding engine. The `mujoco` adapter runs on the mjbatch runtime and the `drake` adapter on the DrakeUni runtime, so cite those alongside the original engines:

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

## License

Apache-2.0; see [LICENSE](LICENSE).
