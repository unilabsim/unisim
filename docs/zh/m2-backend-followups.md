# M2 其余后端评估

[English](../en/m2-backend-followups.md) | [中文](m2-backend-followups.md)

本文记录 #108/#112 工作包 E 的源码评估。Roadmap 要求 MuJoCo、MJWarp、IsaacGym、IsaacSim 真实实现，并要求其余五后端具有具体评估与后续项。以下提案不扩大该实现承诺，也不把明确拒绝计为功能交付。本次评估未运行原生引擎测试。

## 证据与后续动作

| 后端 / 后续项 | 源码证据 | 首个需要验证的实现 | 尚未解决的边界 |
| --- | --- | --- | --- |
| Genesis 1.3.3 — [#120](https://github.com/unilabsim/unisim/issues/120) | 已有公共 `Scene.add_entity(morph=[...])`；heterogeneous loader 校验 joint 名称/类型/DoF 和 variant 惯性。UniSim 当前绑定单 entity。 | 独立实体映射及原生同布局 heterogeneous target；被动关节不占 action。 | 原生 assignment 为 balanced blocks，例如 N5/K2 `[0,0,0,1,1]`；任意指派需上游公共 API。虽然 loader 代码更宽，多 link variants 仍需 pinned runtime 证据。 |
| Motrix 0.8.2 — [#121](https://github.com/unilabsim/unisim/issues/121) | 发布 stubs 提供 `World.attach`、批量 SceneData 和逐 data Link mass/COM override。UniSim 当前局部 reset 会重置整个选中 world。 | 组合独立具名根，冻结全部四元数/状态映射，实现不重置整 world 的 masked entity writes。 | 逐环境 geometry 和一致惯量变化尚未验证；mass/COM override 不等于几何 variants，需与 SDK owner 协作。 |
| Drake / drake-uni — [#122](https://github.com/unilabsim/unisim/issues/122) | Runtime 接收一个模型路径；已检查 C++ 要求 parser 单 model instance，但已有多个 free-joint 映射。 | 先测试一份组合 MJCF 中多个独立根与被动关节，再绑定 UniSim 实体地址。 | 单 model instance 不代表单 root。Geometry variants 和多 model instance 可能需上游显式 runtime 契约，不能访问私有 plant 绕过。 |
| Newton 1.5.1 — [#123](https://github.com/unilabsim/unisim/issues/123) | `ModelBuilder.add_builder` 与 world 构建可组合实体；当前 adapter replicate 单模板且假定首个 free root。 | 每 world 组合 sub-builder，audit body/root/joint/actuator 映射，先支持 fixed 和多个 free root 再声明 variants。 | SolverMuJoCo 转换、ArticulationView 和逐 world 不同几何需真实运行证据。Replication 本身不证明 heterogeneous 支持。 |
| SuperDex — [#124](https://github.com/unilabsim/unisim/issues/124) | 当前冷路径 ModelPlan 和 batch executor 每 world 绑定一个 articulation actor；MJCF audit 拒绝多个 free root。 | 为多个 actor 扩展 audited plan 与公共 executor metadata；先用 serial 原生参考再验证 batch 一致。 | 安装的 `superdex-*-uni` 扩展 batch ABI 必须绑定精确源码/构建。公共项目 main 不能单独证明该分发 executor 的多 actor 能力。 |

## 源码来源

UniSim adapter 评估基于开发提交 `0189060b7eb9cf441bf43a818102856bbb3013c9`，main 为 `dd8984c28a5042d267600c95cb403334f112f193`。后续核心后端集成不代表这五个 owner 模块已支持新能力。各链接 issue 包含具体 UniSim 路径、上游 owner 接口及拟议验收。

Genesis 上游 v1.3.3 固定在 `76f8f5b3457e7c6d6a078de2244066f9a8694c45`；heterogeneous loader 和测试仅为源码证据，不是运行结果。Drake runtime 源码检查于 `4cdc9ba4c9b1a7542755631afe0d57dbb54cdb63`，与安装版本的对应仍需验证。Motrix 证据来自安装的 0.8.2 公共 stubs。SuperDex 公共源码检查于 `b717b1ccf8a9312ac63e709bd0bead32f37fdd6f`；扩展 executor 分发来源仍是显式依赖。

## 责任与验收

各后续项由 UniSim adapter owner 负责，缺失的公共 runtime 接口归上游。UniLab 消费公共状态/场景契约，不能连接引擎私有 actor。实现前确认 PR base 和具体支持组合；新增上游协议、长期支持或 CI 承诺须在 issue/ADR 明确决策。

验收必须确认实际实例身份、源/原生惯量与几何、状态/动作布局、被动 articulation、选行/实体保留、reset 跨后续 step 的持久性、镜像隔离和完整回放。记录精确 runtime/版本/设备/命令证据。同布局或受限 assignment profile 可诚实拒绝其它组合；任何 profile 不能只因构造守卫或 mock 测试通过就宣称支持。

Genesis 原生探针应使用独立进程，在引擎初始化前固定设备可见性。仓库的 stub 设备测试现已恢复原始环境，避免隐式隐藏 GPU 0、使后续 CUDA 测试变成跳过。此测试隔离修复本身不验证 Genesis 多实体执行。
