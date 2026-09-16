# M1 设计决策：语义能力与导入证据

[English](../en/adr-m1-capabilities.md) | [中文](adr-m1-capabilities.md)

## 背景与决策

Issue #91 M1 将 adapter 安装、支持语义、验证状态和实际采用设置分开。依赖顺序为 #101 公共维度 → #102 限定范围的证据 → #103 构造报告 → #104 显式请求校验 → #105 清单和真实运行验收。源码审查和 fixture 可以并行，验收则依赖最终共享契约。

`CapabilityReport` 包含 `CapabilityScope` 和不可变 `CapabilityDeclaration` 记录。每项声明包含点分 feature key、`SupportLevel`（`exact`、`approximate`、`unsupported`、`unknown`）、原因、可选 `CapabilityCondition` 条件和独立 `CapabilityEvidence`。`get_adapter_capabilities(name, profile="default")` 读取源码声明，不导入或发现 SDK。缺失特性和不满足条件的查询返回 unknown。`SimBackend.get_capabilities()` 从既有权威 API 聚合 DR/play/fixed-variant 声明，不维护另一份注册表。

## 证据与版本匹配

证据记录区分源码审查和运行结果。只有 adapter、profile、UniSim 版本、adapter 版本/commit、engine 版本、平台和设备全部明确且完全匹配、runtime availability 显式为 true 时，运行结果才算验证。未知值为 `None`；缺 SDK、身份字段缺失、失败/跳过、版本/profile 不匹配以及仅源码证据均不算真实运行已验证。近似实现在通过实测后仍然是近似。构造成功不会自动产生广泛验证声明。

新增证据时记录精确命令、小资产、revision（含工作区修改状态）、runtime/SDK、设备、数值容差和结果，再仅对该范围实际执行的特性附加 `CapabilityEvidence(kind="runtime", ...)`。更新时新增范围限定的记录，不扩大版本范围或静默复用旧证据。撤销时保留来源并设置 `revoked=True`。初始清单仅保留固定版本源码审查；诊断 JSON 是可审查证据，不会自动升级声明。

## 配置报告与生命周期

`SimBackend.get_import_report()` 返回 construction/materialization 阶段缓存的不可变 `ImportReport`。其 `ConfigurationField` 记录 requested/effective 值、difference（`exact`、`overridden`、`approximate`、`unknown`、`not_applicable`）、provenance、单位、坐标系及 `ConfigurationScope`（实体、环境 ID 和 variant）。源声明、引擎回读、adapter 设置和未验证值保持区分。缺失 effective 值绝不由源值填充。solver/integrator 名称保留引擎自身含义，同名不承诺数值等价。`to_dict()`/`from_dict()` 往返不含 SDK 对象。

报告覆盖 solver、integrator、步长、重力、actuator 映射、碰撞过滤、质量/惯量和 sensor。MuJoCo 回读已编译模型设置；MJWarp 区分 host source 与实际 device options。Isaac worker 返回带版本及自身来源的配置 envelope；host XML 不是 worker 回读。无法回读的项目保持 unknown。canonical/variant 行按范围记录，不从环境 0 推广到全体环境。报告是初始快照：reset DR 后的当前逐环境值仍以既有属性查询为准。读取报告不重新解析资产或推进物理。

## 校验与兼容性

`SemanticRequirements` 选择 feature key、报告 setting key、profile、条件及逐项授权的近似 key。`validate_semantic_requirements()` 不加载 SDK 即可校验声明/报告。`create_backend(..., semantic_requirements=...)` 执行声明预检、完成 materialization、将条件绑定到真实配置并校验请求字段后返回。未知/不可用语义和未授权近似按 fail-closed 原则拒绝，诊断包含 backend/profile/field；校验失败时关闭已初始化资源。显式 factory 步长等覆写在报告中保持可见。固定 worker 重力与源声明冲突时需要近似授权；无法比较的 native solver/integrator 名称保持 unknown，不能通过严格 setting 要求。`require_runtime_verified=True` 额外要求匹配的已附加运行证据，只有源码证据的清单无法满足。

未指定 semantic requirements 的调用者保留原生命周期及 adapter 审计。显式严格路径用于渐进迁移，不代表所有旧导入路径已经完整审计。严格构造已经 materialize backend，不应再次调用 `materialize()`。既有 SuperDex 近似 opt-in 和 IsaacSim 接触拒绝仍然有效。没有引擎回退、solver 自动替换、新 importer IR、runtime SDK 依赖或热路径 XML 解析。

提供导入报告时，独立 validator 和 factory 都会将配置条件与所有适用范围的 effective 字段核对。adapter 参数通过 adapter 自身报告字段提供；通用校验不探测后端私有状态，也不将调用者 kwargs 当作实际采用值。物化前执行 Manager-Based startup 事件的消费者应保留该顺序，在普通物化之后校验，而不是选择提前严格构造。见[路径消融与归属审计](m1-ablation.md)。
