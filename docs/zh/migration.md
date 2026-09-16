# UniLab 迁移

[English](../en/migration.md) | [中文](migration.md)

迁移按后端分阶段进行。每个适配器子任务同时迁移实现和文档，添加可选依赖诊断与一致性覆盖，并更新 UniLab 消费边界。历史上的 `unilab.base.backend` 再导出 shim 已移除；现在只有一个由 `unisim-core` 拥有的生产实现。

MuJoCo 是第一个进程内适配器。它接受包中立的 `SceneCfg`，在构造时物化 XML，并通过 `unisim.SimBackend` 暴露缓存数值状态；任务拥有的场景组装仍留在 UniLab。它的原生批处理执行器是 mjbatch（`unilabsim/mjbatch_uni`，`kevinzakka/mjbatch` 的维护 fork）；不支持异构模型变体，字段级域随机化通过 mjbatch `expand` 与 `set_const` 完成。

Motrix 是第二个进程内适配器。它在相同的公共状态、控制和重置契约之后使用 Motrix 的批处理 `SceneData` 与掩码数据切片。

其余 UniLab 身份在 UniSim 中都是一等适配器：Drake、MJWarp、Genesis、Newton、SuperDex、IsaacGym 和 IsaacSim。后两者复用 `unisim.backend.subprocess_ipc`，在不向宿主进程导入 Kit 或 Python 3.8 模块的情况下解析厂商 worker。缺失 SDK 会在构造时报告；任何后端都不会被静默降级为另一个引擎。

运行时拥有的缓存和 worker 安装使用 `UNISIM_*` 环境变量与 `~/.cache/unisim` 默认值。旧 `UNILAB_*` 名称只作为迁移回退接受，使既有安装可以在不丢失缓存状态的情况下迁移。

## M1 语义要求

既有构造保留原生命周期与 adapter 审计。迁移任务时，先查看 `get_adapter_capabilities("mujoco")`，再显式请求任务所需语义特性和实际配置字段。严格构造返回前已经完成 `materialize()`，因此应移除这一路径上的独立 materialization 调用。请求未知或不支持特性均会 fail-closed；近似授权仅针对具体 key，不会跳过其他 adapter 检查。

```python
from unisim import SemanticRequirements, create_backend, get_adapter_capabilities
from unisim.scene import SceneCfg

static = get_adapter_capabilities("mujoco")  # No SDK discovery or import.
backend = create_backend(
    "mujoco", SceneCfg(model_file="robot.xml"),
    semantic_requirements=SemanticRequirements(
        features=("asset.mjcf", "actuator.motor"),
        settings=("dt", "gravity", "body_mass"),
    ),
)
initial_configuration = backend.get_import_report().to_dict()
```

使用 `backend.get_capabilities()` 聚合既有 DR/play/variant 权威来源。导入报告是初始配置快照，不是 reset randomization 后的当前值。`require_runtime_verified=True` 会拒绝只有源码的证据；SDK 存在或构造成功不会自动验证每项特性。配置条件必须匹配实际报告值，包括 adapter 记录的参数；虚构上下文无法授权另一个 profile。精确边界见 [ADR](adr-m1-capabilities.md) 和[运行证据](m1-runtime-evidence.md)。

对于物化前执行 startup 事件的 Manager-Based 消费者，应保留普通构造及既有 startup/materialization 顺序。物化后、步进前调用 `validate_semantic_requirements(backend.get_capabilities(), requirements, backend.get_import_report())`。该公共 validator 执行与严格 factory 构造相同的配置条件检查，不将 startup 事件移过物化边界。初始报告不能替代 DR 后的当前属性查询。[消融审计](m1-ablation.md) 记录了这一边界及实测报告开销。
