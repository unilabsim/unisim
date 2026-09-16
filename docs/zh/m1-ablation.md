# M1 路径消融与归属审计

[English](../en/m1-ablation.md) | [中文](m1-ablation.md)

## 范围与复现

本次审计比较初始 M1 实现 `b0f77abb9866c2e72b082f5897871b0e1a8b3052` 与优化后的报告路径。[机器可读证据](../evidence/m1-ablation.json) 记录被测 revision、工作区状态、Python 版本、设备、数据形状及耗时。这些测量隔离冷路径报告开销，不证明 rollout 吞吐量，也不扩大[运行支持证据](m1-runtime-evidence.md)的范围。

```bash
uv run --no-sync python scripts/benchmarks/m1_report_ablation.py --baseline b0f77ab --cuda --mujoco --output /tmp/unisim-m1-ablation.json
```

基准从 Git 读取历史模块进行 A/B 比较；可选真实运行路径需要 MuJoCo 开发 extra 和可用的 Warp CUDA 安装。它断言报告与能力声明的序列化值一致。MuJoCo 实验分别运行关闭报告、开启报告和严格 factory 构造，使用 32 个环境及 128 个关节，并在步进后检查最终 qpos/qvel 逐位一致。关闭报告仅通过基准中的临时 patch 实现，不新增生产绕过开关。

## 审计确认的改动

- 配置比较原先为相等判断构造冻结树，再为保存结果重复构造。现在相等的 JSON 值不再产生废弃树，同时保留校验和脱离源对象的保证。同一个 requested/effective 输入只冻结一次。能力序列化不再重复递归序列化证据。
- MJWarp 报告捕获避免从设备复制不使用的环境行，并在发布不可变快照后释放临时 requested 值表。MuJoCo 仅覆写 actuator 时只读取 actuator 表，不再收集完整模型报告。
- 子进程报告一次分组环境 assignment，并将来源与语义规范化合并为一次字段替换。保留 worker 逐环境证据，源 XML 不会替代引擎回读。
- 通用 binding 现在通过公共报告校验配置条件。MuJoCo 和 SuperDex 在各自 adapter 报告中记录已消费参数；factory 不再探测 MuJoCo 私有状态，也不再将转发 kwargs 当作证据。独立 validator 现在会拒绝与物化值冲突的条件，包括混合 variant 范围和未授权近似。
- 重复的 `wrench.body_force` key 改用权威 `dr.interval.body_force` key。含糊的 `state.refresh` key 删除，改用 `state.final_refresh` 和 `state.callback_refresh`。既有 DR/play/variant API 继续作为支持声明的来源。

## 测量记录

环境为 Python 3.13.14 和 NVIDIA GeForce RTX 4090。微基准报告九次重复的中位数；MuJoCo 路径在预热后测量七次。JSON 保留完整精度。

| 隔离操作 | 优化前 / A | 优化后 / B | 含义 |
| --- | --- | --- | --- |
| 配置比较，独立且相等的 `(32, 512, 3)` 值 | 31.461 ms | 16.143 ms | 序列化报告相同，保留独立快照 |
| 配置比较，共享输入对象 | 31.522 ms；Python 峰值 2,376,912 字节 | 7.936 ms；Python 峰值 1,191,512 字节 | 脱离输入的不可变快照，仅相同输入对象共享存储 |
| 能力序列化 | 0.077 ms | 0.031 ms | 序列化声明值相同 |
| CUDA `(4096, 128, 3)` 回读，仅消费一行 | 0.403 ms，复制所有行 | 0.021 ms，复制选中行 | 隔离传输机制，不是完整初始化耗时 |
| MuJoCo 构造 + 物化 | 关闭报告：7.862 ms | 开启报告：9.864 ms；严格路径：10.351 ms | 报告存在可测的冷路径成本 |
| MuJoCo 十个子步 | 关闭报告：0.285 ms | 开启报告：0.281 ms；严格路径：0.285 ms | 状态一致，计时差异不构成吞吐量结论 |

Python 峰值字节来自 `tracemalloc`，不包含 native 和 GPU 分配。报告形状与代表行数量影响传输收益。这些有限测量不证明高 variant 数量下的性能或 Isaac worker 启动速度。

## 归属与生命周期边界

上级 `AGENTS.md` 要求后端专有行为保留在 `SimBackend` 边界之后，DR 仍由 Manager-Based event terms 负责，资产解析留在冷路径。本次改动将参数解释保留在 adapter，将通用条件检查保留在既有 validator。不向基础包加入环境/训练策略、第二套 DR 注册表或引擎依赖。

UniLab 的 Manager-Based 环境先执行 startup 事件，再调用 `materialize()`。严格 factory 构造有意提前完成物化，因此不能普遍直接替换这套生命周期。存在 startup 工作的消费者应保留普通构造，执行 startup 事件及既有物化流程，然后在步进前调用 `validate_semantic_requirements(backend.get_capabilities(), requirements, backend.get_import_report())`。报告仍描述初始采用配置；DR 后当前值必须通过当前属性查询获得。本次不新增物化幂等兼容层，也不修改下游生命周期。

Focused 回归覆盖独立 validator 调用、近似授权、配置范围、脱离源对象的快照及选中设备行。仓库的 `make check` 与 `make package` 仍是完成门禁；微基准耗时是诊断证据，不是 CI 性能阈值。
