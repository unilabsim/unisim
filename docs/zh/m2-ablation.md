# M2 路径消融与契约审计

[English](../en/m2-ablation.md) | [中文](m2-ablation.md)

## 范围与复现

[机器可读记录](../evidence/m2-ablation.json) 对比 UniSim 基线 `e952419` 与干净实现 `fcbf8c7`，以及 UniLab 基线 `044a11ff` 与干净实现 `71c11431`。每项 A/B 使用相同输入并核对输出相等。宿主 tensor double 隔离映射或传输量，不代表原生仿真或训练吞吐。必需的原生回归门禁在 PR 中单独记录。

```bash
uv run --no-sync python scripts/benchmarks/m2_path_ablation.py --output /tmp/gym-ab.json
uv run --no-sync python scripts/benchmarks/m2_entity_query_ablation.py --output /tmp/query-ab.json
uv run --no-sync python scripts/benchmarks/m2_sim_reset_ablation.py --output /tmp/sim-ab.json
uv run --no-sync python scripts/benchmarks/issue141_fk_path_ablation.py --output /tmp/issue141-fk-ab.json
# 在 UniLab 消费端 checkout 中：
uv run python scripts/benchmark/physics/m2_reset_ablation.py --output /tmp/reset-ab.json
```

脚本从 Git 加载旧实现，不用手写近似替代。Query A 同时加载旧共享 snapshot helper。UniLab 增加只改变行索引的第三条路径，以区分索引和无用快照的影响。不增加生产 A/B 开关或 CI 性能阈值。

## 发现与修改

- Gym 原先通过 Python 逐环境、逐关节重建公共状态，每次刷新重新计算 body mask。现在冷路径绑定真实原生地址并批量 gather，保留 pending indexed reset、无归属 body 槽、COM 转换和旧接口映射。
- MuJoCo/MJWarp 单实体查询原先先构造全部实体 root、复制无关 joints，再取请求的 snapshot。现在只 gather 一个实体；高级索引已分配的数组不再重复 copy，结果仍与内部缓存隔离。
- Subprocess getter 使用冷路径绑定的实体、名称和列映射，不再扫描 layout 或重复复制 gather 结果。
- IsaacSim 稀疏 joint reset 原先下载全部环境的 joint position/velocity 后再选择，还给未修改实体上传 ID。现在先在设备上选择再下载，并跳过未修改实体。原生 setter 值和顺序、actuator reset/update 及提交后 refresh 保持不变。
- 两个 MuJoCo 系 reset adapter 复用已准备的请求绑定。内部 reset-impact owner 在冷路径一次绑定后代 body、DoF、actuator 和 activation 清理地址，不再在 reset 时遍历静态拓扑或读取模型 activation 元数据。
- UniLab reset 暂存以行映射替换平方级查找，只保存请求字段。只有合并不同 joint 字段选择、需要补齐缺失列时才读取当前状态。冷路径拒绝将后代 body 作为逻辑 root，防止 root 读取、默认值和写入不一致。
- #141 后续路径对每个 legacy MJCF 源只解析一次并同时产出 metadata 与 FK 表；worker 只准备一次 FK arrays，fixed-variant 分组改用 NumPy 行索引。逐环境稀疏 overlay 保留：常驻 dense buffer 已测试并拒绝，因为其消耗全批内存且没有稳定 refresh 收益。

## 测量与边界

| 隔离操作 | A | B | 证据边界 |
| --- | --- | --- | --- |
| Gym refresh，N=4096，32 joints | 142.329 ms | 1.387 ms | 全部共享槽逐元素相等；不含 GPU/IPC |
| MuJoCo 实体查询，N=4096 | 714.07 µs | 128.90 µs | Root/joint 数组相等；仅宿主缓存 |
| MJWarp 实体查询，N=4096 | 679.83 µs | 124.55 µs | Root/joint 数组相等；无设备传输 |
| IsaacSim 单 joint reset，N=1024，选一行 | 下载 262,144 字节 | 下载 8 字节 | 执行 tensor double 统计字节；原生调用相同 |
| UniLab pose 暂存，N=R=4096 | 23.456 ms | 仅行映射：1.697 ms；稀疏字段：1.407 ms | 相同的单次 reset 请求；不含 engine/IPC |
| UniLab defaults 暂存，N=R=4096 | 23.876 ms | 仅行映射：3.243 ms；稀疏字段：1.877 ms | 当前快照消除：557,056 → 0 字节 |
| #141 冷 metadata+FK 扫描，128 bodies | 1.002 ms / 峰值 313,765 字节 | 0.817 ms / 峰值 208,935 字节 | metadata/FK 表相等；本地 NumPy 文件中位数 |
| #141 stage+refresh，N=512、128 bodies、4 variants | Stage 38.559 ms；refresh 6.343 ms | Stage 37.826 ms；refresh 6.348 ms | 全部发布槽相等；不含 SDK/GPU/IPC |

Gym 批量 gather 用临时内存换取速度：N=4096 时跟踪到的 Python/NumPy 峰值分配从 2.25 MB 增到 3.54 MB；冷路径原生索引缓存也随 N、joint/body 数增长。这不是原生/GPU 内存测量。Query 峰值分配从 CPU 的 2.72 MB、Warp 的 2.08 MB 降至 0.95 MB。时间是本地中位数，受负载影响，不是速度保证。

MJWarp 局部 reset 仍有全批状态保留屏障。独立真实 CUDA 探测使用一个 free body、无接触：N=512 选一行约 0.334 ms，上传 90,624 字节、读取 49,152 字节 persistent channels；选全部行约 0.364 ms，传输量相同。JSON 保存了完整探测源码。这是仍然存在的规模成本，宿主优化未消除它。替换该路径需要设备端 selected scatter 及 warmstart/control/wrench/sensor 隔离证据；直接删除保留屏障会违反 reset 契约。IsaacSim 提交后的全量 refresh 同样保留为显式成本。

## 契约审计与保留路径

父级 `AGENTS.md` 要求后端行为位于 `SimBackend` 后、资产元数据留在冷路径、DR 由 Manager-Based event 拥有，并使用可 pickle 的环境工厂。本次移除 reset 中的静态拓扑解释并修正物理 root 绑定。UniLab 继续只消费公共 state/layout/reset 方法；不引入引擎私有调用、热路径 XML 解析、任务名称分派、第二 DR 生命周期、learner 依赖或新 `utils` owner。Dict 观测、动作维度和已有 sim2sim policy-I/O 保持不变。

整体环境 reset 与局部实体 patch 保留不同意图，经同一个原生 owner 提交。兼容映射保留旧 wire 形状；独立原生/源身份 audit 仍必要。Scratch/full-forward 策略、原生失败封锁和 sensor/wrench 生命周期保留，因为删除它们会改变语义，并非消除冗余。

维护者于 2026-09-17 明确允许将 IsaacSim 录制验收延期至 [#133](https://github.com/unilabsim/unisim/issues/133)。该相机 profile 仍为未验证，延期不把启动失败变成渲染通过。M2 完成仍需满足 #108/#113 记录的最终集成、打包和下游依赖门禁。
