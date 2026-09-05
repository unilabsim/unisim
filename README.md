# UniSim

[English](README.md) | [中文](README_zh.md)

UniSim provides backend-neutral physics contracts and optional engine
adapters for robot learning and simulation. The PyPI distribution is
`unisim-core`; the Python import namespace is `unisim`.

A single `SimBackend` contract covers state access, control, reset, and
domain-randomization boundaries, so the same task code runs on MuJoCo,
Motrix, Drake, MJWarp, Genesis, IsaacGym, or IsaacSim without engine-specific
branches. The base install depends only on NumPy; every engine SDK is an
optional extra loaded lazily, and importing `unisim` never imports an engine.

## Relationship to UniLab

UniSim is the extracted, backend-neutral physics layer used by UniLab.
UniLab retains Hydra configuration,
task/env/manager lifecycle, robot assets, RL training, checkpoints, and
sim2sim policy I/O; UniSim owns the physics contract, adapter lifecycle and
state translation, optional-runtime diagnostics, and the shared subprocess
IPC layer. There is exactly one production implementation of each backend,
owned by this repository — UniLab only assembles task-owned scene and
configuration inputs and consumes the public contract. UniSim never imports
UniLab.

## Installation

```bash
pip install unisim-core                # base: contract, factory, fake backend
pip install "unisim-core[mujoco]"      # plus an engine extra when needed
```

Available extras: `mujoco`, `motrix`, `drake`, `mjwarp`, `genesis`, `newton`,
`isaacgym`, `isaacsim`. The Isaac extras are empty spellings because those
vendor SDKs are not redistributable; their adapters discover dedicated worker
installations at construction time. See
[`docs/support-matrix.md`](docs/support-matrix.md) for the full adapter
support matrix.

The `newton` extra is isolated from `mjwarp`: it pins Newton 1.5.1 with the
MuJoCo-Warp 3.11.0 line, while `mjwarp` uses 3.10.0.3. Install only one of
these two extras in an environment.

## Quick start

The public boundary is deliberately lazy and safe to import anywhere:

```python
from unisim import SimBackend, create_backend
```

Construct a backend through the factory with a package-neutral `SceneCfg`:

```python
backend = create_backend("mujoco", scene=scene_cfg, num_envs=64, sim_dt=0.01)
backend.materialize()      # cold path: parse XML, build engine objects
backend.reset()
state = backend.get_state()
backend.step(ctrl)         # hot path: validated arrays, cached handles
```

Each adapter fails closed with an actionable, backend-specific diagnostic
when its optional runtime is missing — no backend is silently downgraded to
another engine. Engine-native model and data objects never escape the
adapter; state and control flow through validated NumPy arrays.

A deterministic `FakeBackend` and the `assert_backend_conformance` helper let
consumers test task code without any engine installed. `BenchmarkCase` and
`BenchmarkResult` are reserved schema extension points for a future benchmark
package; no workload runner is implemented here.

External worker roots can be configured with `UNISIM_ISAACGYM_HOME`,
`UNISIM_ISAACGYM_PYTHON`, `UNISIM_ISAACSIM_HOME`, and
`UNISIM_ISAACSIM_PYTHON`. The package also accepts the former `UNILAB_*`
spellings as a migration fallback.

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — ownership boundary between
  UniSim and UniLab
- [`docs/support-matrix.md`](docs/support-matrix.md) — adapter install and
  runtime requirements
- [`docs/migration.md`](docs/migration.md) — migrating from the historical
  UniLab backend layer
- [`docs/benchmark-api.md`](docs/benchmark-api.md) — reserved benchmark
  schemas
- [`docs/release.md`](docs/release.md) — TestPyPI and automated production
  release procedure

## Development

```bash
make sync       # locked environment plus the MuJoCo test extra
make check      # Ruff + pytest
make package    # source distribution and wheel for local inspection
```

Every adapter must document its supported Python/platform/runtime matrix and
pass the conformance helper before it is published.

## Citation

If UniSim contributes to your research, please cite the UniLab paper:

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

### Physics backends

When you use a specific backend through UniSim, please also cite the
corresponding engine. The `mujoco` and `drake` adapters build on the MuJoCoUni
and DrakeUni runtimes, so cite those alongside the original engines:

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

% MuJoCoUni (runtime of the `mujoco` adapter)
@article{jia2026mujocouni,
  title   = {MuJoCoUni: Persistent Batched Runtime Primitives for MuJoCo},
  author  = {Jia, Yufei and Wu, Junzhe},
  journal = {arXiv preprint arXiv:2605.24922},
  year    = {2026}
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

## License

Apache-2.0, see [LICENSE](LICENSE).
