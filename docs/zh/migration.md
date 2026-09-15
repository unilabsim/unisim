# UniLab 迁移

[English](../en/migration.md) | [中文](migration.md)

迁移按后端分阶段进行。每个适配器子任务同时迁移实现和文档，添加可选依赖诊断与一致性覆盖，并更新 UniLab 消费边界。历史上的 `unilab.base.backend` 再导出 shim 已移除；现在只有一个由 `unisim-core` 拥有的生产实现。

MuJoCo 是第一个进程内适配器。它接受包中立的 `SceneCfg`，在构造时物化 XML，并通过 `unisim.SimBackend` 暴露缓存数值状态；任务拥有的场景组装仍留在 UniLab。它的原生批处理执行器是 mjbatch（`unilabsim/mjbatch_uni`，`kevinzakka/mjbatch` 的维护 fork）；不支持异构模型变体，字段级域随机化通过 mjbatch `expand` 与 `set_const` 完成。

Motrix 是第二个进程内适配器。它在相同的公共状态、控制和重置契约之后使用 Motrix 的批处理 `SceneData` 与掩码数据切片。

其余 UniLab 身份在 UniSim 中都是一等适配器：Drake、MJWarp、Genesis、Newton、SuperDex、IsaacGym 和 IsaacSim。后两者复用 `unisim.backend.subprocess_ipc`，在不向宿主进程导入 Kit 或 Python 3.8 模块的情况下解析厂商 worker。缺失 SDK 会在构造时报告；任何后端都不会被静默降级为另一个引擎。

运行时拥有的缓存和 worker 安装使用 `UNISIM_*` 环境变量与 `~/.cache/unisim` 默认值。旧 `UNILAB_*` 名称只作为迁移回退接受，使既有安装可以在不丢失缓存状态的情况下迁移。
