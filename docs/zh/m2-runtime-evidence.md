# M2 运行证据与剩余门禁

[English](../en/m2-runtime-evidence.md) | [中文](m2-runtime-evidence.md)

[机器可读记录](../evidence/m2-runtime.json) 对应干净实现提交 `2e571e31c835592becb844fb368150a427e3e34b`。集成合并 `e30ee8379a0a16abcf1cf4e444e1efe0ad8f74ed` 的文件树相同（`git diff` 为空）。此次证据新增只改测试/文档，不改写来源。[#108](https://github.com/unilabsim/unisim/issues/108) 仍开放：物理测试通过不代表全部发布或渲染门禁已完成。

## 物理与协议门禁

```bash
uv sync --locked --extra mujoco --extra mjwarp
UNISIM_TEST_ISAACGYM_SCENE=1 UNISIM_TEST_ISAACSIM_SCENE=1 make check
```

结果：**853 passed，11 skipped**，106.63 秒。Ruff、mypy、Pyright 通过，包含真实 MuJoCo CPU、MJWarp CUDA、两个隔离 vendor worker、公共 factory-to-worker 场景及旧 IsaacGym model-file 输入。可选 runtime 跳过不计为原生通过。首次新 audit worktree 缺 MuJoCo extra 导致收集失败；按上述命令安装明确 extras 后完整重跑，解决环境错误。

| Runtime | 已验证行为 | 边界 |
| --- | --- | --- |
| MuJoCo 3.11 / mjbatch 0.2.1 | 多根/被动关节、N2/K2 和 N5/K2 身份、编译惯性 oracle、局部 reset、独立原生 rollout 和完整 mocap 回放 | 已声明 MJCF 组合子集；旧 whole-model 兼容不等于新公共实体声明 |
| MuJoCo-Warp 3.11 / Warp 1.16 | 逐 world 身份/默认值、GPU 状态/坐标 oracle、reset 通道保留、故障处理、main-world/scratch 路由及独立单 world rollout | 局部 reset 使用文档化 forward barrier，不代表新增 selective-forward 性能承诺 |
| IsaacGym Preview 4 / Python 3.8 | 原生资产身份、质量/COM/惯量、实体 actuator-only 控制、非 round-robin variants、连续局部 reset、零 DoF 和旧 wire 兼容 | 实体查询用 link 原点速度；旧 wire 经显式映射保留 native COM 速度约定 |
| IsaacSim 5.1 / IsaacLab 0.47.2 / Python 3.11 | 实际 prim/view 身份、固定/浮动受控与被动 articulation、局部 reset、镜像隔离及公共 factory/default controls | 已支持 same-drive round-robin profile；未实现格式/布局拒绝；原生相机仍未通过 |

JSON 记录 fixture 源码哈希、宿主依赖版本、GPU/driver 和命令。数值期望由对应测试定义：独立编译/原生值和同后端 rollout 使用各自容差，不逐位比较跨引擎复杂接触轨迹。单独的旧 IsaacSim diagnostic 也已通过统一 runtime 的真实初始化、reset 和 step。

## 原生渲染

IsaacGym 生成了非均匀 320×240 RGB 图像。Capture 保留 qpos/qvel/ctrl 和被指派完整场景身份。None/record 轨迹最大绝对差为零，包含位于桌面外 z=0.03 的物体：两次均下落到约 0.0217596，检查录制没有引入地面。仓库可复现回归为 `tests/contract/test_scene_rendering.py`，通过 `UNISIM_TEST_ISAACGYM_RENDER=1` 启用。

本机 IsaacSim 相机初始化在创建场景前失败。仅运行 stock IsaacLab AppLauncher、启用 `enable_cameras=True`，不导入 UniSim/资产，也在 RTX/Hydra 场景初始化中 SIGSEGV（-11）。分别禁用用户配置加载/持久化、Fabric scene delegate 和 sampled direct lighting 均未解决。[#133](https://github.com/unilabsim/unisim/issues/133) 记录独立复现。对应渲染测试仅在 `UNISIM_TEST_ISAACSIM_RENDER=1` 时运行；它未包含在已通过物理门禁中这一事实明确保留，不构成原生相机支持声明。

## 下游与发布边界

[UniLab #1599](https://github.com/Motphys/UniLab/issues/1599) 和 [draft PR #1600](https://github.com/Motphys/UniLab/pull/1600) 用注册的可 pickle EnvFactory 消费公共实体契约。开发验证通过 1477 测试、70% 覆盖率、必需的 34/34 module 与 35/35 script benchmark import 检查，以及两个真实 IsaacSim consumer 场景。这些运行使用明确标识的 editable M2 UniSim checkout。已发布依赖锁仍需约定的上游版本与最终安装/CI 验证，不用 editable 结果替代该门禁。

其余五后端评估和后续 issues 见[责任矩阵](m2-backend-followups.md)，不计为五个额外实现。关闭 #108 前需保留最终 head 的打包/CI 证据，解决或明确决策原生渲染验收边界，并完成下游已发布依赖门禁。本文不把缺失证据提升为成功。
