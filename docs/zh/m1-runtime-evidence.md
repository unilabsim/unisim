# M1 运行证据与剩余验收

[English](../en/m1-runtime-evidence.md) | [中文](m1-runtime-evidence.md)

## 范围与复现

2026-09-16 的初始运行使用基于 `6d62d9ba40a1f637fb04da190fd38f64d02b05d1` 的实现工作区（dirty，不代表未修改基线）。[机器可读记录](../evidence/m1-runtime.json) 保留 revision、dirty 状态、精确命令、host/worker 版本、fixture 哈希、设备、容差、逐项检查及报告样本。升级验证声明前应针对已提交实现重跑。mock 和 skipped 测试均不计为真实 runtime 通过。

```bash
uv run --no-sync python scripts/diagnostics/check_support.py --runtime mujoco --output /tmp/m1-mujoco.json
uv run --no-sync python scripts/diagnostics/check_support.py --runtime mjwarp --output /tmp/m1-mjwarp.json
uv run --no-sync python scripts/diagnostics/check_support.py --runtime isaacgym --output /tmp/m1-isaacgym.json
uv run --no-sync python scripts/diagnostics/check_support.py --runtime isaacsim --output /tmp/m1-isaacsim.json
uv run --no-sync python scripts/diagnostics/check_support.py --check-docs
```

独立小资产为 `tests/contract/fixtures/m1_semantics.xml`（motor）和 `m1_position.xml`（position drive）。各自包含 free root、一个 hinge、显式质量/惯量、自碰撞 exclude、gyro/accelerometer、Newton/Euler 源选项及重力。源 dt 为 0.004 s、factory 请求为 0.002 s，明确覆盖覆写。诊断使用两个环境并执行真实 reset/step；检查语义字段和单位，不要求跨引擎轨迹相同。

## 已记录运行

| Runtime | 环境 | 结果 | 实际检查与剩余边界 |
| --- | --- | --- | --- |
| MuJoCo CPU | Python 3.13.14；MuJoCo 3.11.0；mjbatch-uni 0.2.1 | passed | Newton/Euler、dt、重力、motor gear、exclude、质量/惯量和 sensor map；实际步进后 gyro/accel 有限 |
| MJWarp CUDA | Python 3.13.14；MuJoCo-Warp 3.11.0；Warp 1.16.0；RTX 4090，driver 595.84 | passed | 相同 fixture 检查、实际 device 步长/重力和真实 CUDA 步进；不承诺广泛 solver 等价 |
| IsaacGym worker | Python 3.8.20；IsaacGym 1.0rc4；Torch 2.4.1；RTX 4090 | passed | worker dt/重力、position gain、自碰撞禁用、逐环境质量/惯量；真实 gyro；拒绝 motor/accelerometer；integrator 未知 |
| IsaacSim worker | Python 3.11.16；IsaacSim 5.1.0.0；IsaacLab 0.47.2；Torch 2.7.0+cu128；RTX 4090 | passed | worker dt/重力、position gain、自碰撞禁用、逐环境质量/惯量；真实 gyro；拒绝 motor/accelerometer；solver/integrator 未知 |

绝对容差为 dt 1e-9 s、重力 1e-5 m/s²、显式质量/惯量 1e-6。sensor 检查要求 `(2, 3)` 数组有限，不声称完成精度标定。reset/step 后缓存快照必须完全不变。worker position drive 和碰撞条目在无引擎回读时保留 adapter-setting 来源；数值检查不会重标来源。不支持 sensor 的拒绝是拒绝证据，不是实现证据。Gym float32 重力回读即使在诊断容差内，也可能被标为 approximate；严格重力要求仍需显式授权。异构 solver 名称保持 unknown，不会被当作普通覆写而静默接受。

## 依赖、owner 与未完成工作

| 工作项 | 依赖 | Owner 边界 | 验收状态 |
| --- | --- | --- | --- |
| #101 声明 | M0 语义 | capability/contract | 已实现无 SDK 契约与实例聚合 |
| #102 证据 | #101 schema | capability/adapter profile | 已实现精确身份匹配、撤销与序列化；只有源码的清单仍未验证 |
| #103 导入报告 | #101 和 #102 | materialization/worker IPC | 已接入 MuJoCo、MJWarp 及两个 worker；不可获取的 effective 字段保持未知 |
| #104 严格校验 | #101–#103 | factory/binding/adapters | 已实现显式严格请求；既有调用保留生命周期及审计 |
| #105 清单与验收 | #101–#104 | conformance/runtime owner | 九项源码条目及四次真实小资产运行；更多 profile 和未知字段仍待完成 |

declared base 为 `main`；集成分支为 `dev/issue-91-trusted-multibackend`。表格记录实现边界，不表示自动关闭 issue 的授权。CPU、CUDA 和 vendor worker 证据保持独立。Motrix、Drake、Newton、Genesis 和 SuperDex 在此进行了源码审查，但没有执行该诊断；真实 profile 验证由对应 adapter/runtime owner 承担，目前仍未验证。

materialization/worker owner 必须补齐未明确的 Isaac solver/integrator 回读，消费者才能严格要求这些字段。adapter/runtime owner 需补充更广碰撞/接触行为及 sensor 精度证据；当前快照验证配置 mask/设置，不覆盖所有物理接触情形。多实体组合和通用接触 API 仍归 #84/#72。缺失 SDK 不会被静默计为通过：runtime 不可用时以失败退出并给出诊断，由对应 runtime owner 成功重跑前，该验收保持未完成。
