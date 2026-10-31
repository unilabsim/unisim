# Ray-Query 插件契约

[English](../en/ray-query.md) | [中文](ray-query.md)

UniSim 定义了 backend-neutral 的 ray-query 插件契约，使 ray caster 实现（例如独立分发的 `uniray` 包）可以在不暴露引擎私有类型的前提下，为任务代码提供批量 ray 查询。契约包含两种传输：用于工具、调试与一致性验证的 NumPy profile，以及接入训练 tensor lifecycle、不绕行 host 的 Torch-native profile。

## 公开接口

以下名称均可从 `unisim` 包根安全地惰性导出，实现位于 `unisim.ray_query`：

- `RayCaster` —— 抽象的插件生命周期：`materialize(scene)` → pose 更新 → trace → `close()`。
- `RaySceneDescription` —— 冷路径消费的 backend-neutral 结构 of arrays（SoA）场景描述。
- `RayCollisionDescription` / `RayMeshDescription` —— detached collision companion，携带 mesh topology、geom/mesh identity、semantic tag 与 body ancestry。
- `RayCasterCapabilities` —— 细粒度的 capability 声明。
- `RayTraceOutputs` / `RayTraceResult` —— 单次调用的输出请求与结果容器。
- `RayTraceFilter` —— 不可变的静态 geom/body/tag filter 请求。
- `RayPoseTensorView` / `RayTraceTensorResult` —— 调用方持有的 Torch pose 输入，以及 caster 持有、可复用的 Torch 结果 view。
- `RayGeomType` —— 几何基元类型（`plane`、`sphere`、`box`、`cylinder`、`capsule`、`ellipsoid`，以及预留的 `mesh`）。
- `ray_geom_type(...)` —— fail-closed 的原生 MuJoCo geom code/名称映射辅助函数。
- `require_ray_trace_outputs(...)` —— fail-closed 的输出协商辅助函数。
- `require_ray_trace_query(...)` —— fail-closed 的静态 filter 与最小距离协商辅助函数。
- `RAY_CASTER_SPECS` / `ray_caster_spec(...)` —— 已声明的插件清单。
- `create_ray_caster(...)` —— 惰性工厂分发。
- `FakeRayCaster` —— NumPy 解析参考实现，额外提供 CPU Torch transport profile。
- `assert_ray_caster_conformance(...)` —— 供插件作者复用的一致性检查。

## 类型边界

host profile 只接受 NumPy；训练 profile 接受 Torch tensor，但不改变包导入行为。`mujoco.MjModel`/`mujoco.MjData`、Warp kernel 或数组、CUDA 指针、stream、event 以及任何其他后端私有类型都不得出现在签名或结果中。真实的 adapter 在自己一侧把原生场景与位姿状态翻译为公开 descriptor 和 tensor。导入 `unisim` 不会导入任何引擎 SDK 或 Torch；插件包由 `create_ray_caster` 惰性导入，tensor 方法惰性导入 Torch，缺少包或入口点时以 `OptionalDependencyError` fail closed。

## 场景描述

`RaySceneDescription` 为每个 geom 携带一条 SoA 记录：`geom_types`、`geom_sizes` `(n, 3)`、`geom_local_pos` `(n, 3)`、`geom_local_quat` `(n, 4)`（单位 `wxyz` 四元数）和 `geom_body_ids` `(n,)`，外加显式的 `num_bodies`。`body_names` 是公开 body 轴；MuJoCo family 中的匿名 body 会获得稳定的 `body_<i>` alias。可选 body-name 选择会选取并重排该轴，同时 remap geom body ID；缺失、重复或空名称 fail closed。geom 在其所属 body 的局部坐标系中表达。尺寸遵循 MuJoCo 风格的基元约定：sphere 为 `(radius, 0, 0)`；box 为半尺寸 `(x, y, z)`；cylinder 与 capsule 为 `(radius, half_length, 0)`，轴向为局部 `+z`；ellipsoid 为半轴 `(x, y, z)`；plane 是无限的局部 `z = 0` 平面，法线为 `+z`，忽略其尺寸。所有数组在构造时校验、拷贝并写保护；非法输入 fail closed。

MuJoCo family 后端通过 `SimBackend.get_ray_scene_description(body_names=None)` 提供该投影，并用 `ray_geom_type` 映射 MuJoCo geom code。投影仅保留 `contype` 或 `conaffinity` 非零的 geom；两个掩码均为零的纯视觉 geom 会被排除。该调用是冷路径元数据；ray 热路径不会解析 model 或刷新 descriptor。Motrix 投影仅覆盖编译几何与原生 runtime 完全一致的已审计 source；生成式 hfield 地形、附加 fragment、逐行 mesh variant，以及异构 fixed-variant source 会 fail closed，而不是静默遗漏或错误表示 runtime 几何。

companion 只覆盖 collision geometry 与 ray query。Public materialized terrain projection、hfield grid metadata 与 subterrain identity 由 #474 单独跟踪。

本阶段 Motrix collision descriptor 仅支持 primitive：source hfield/mesh geometry fail closed，因为 MotrixSim 的归一化与 native mesh introspection 不同于 detached MuJoCo source 投影。

`SimBackend.get_ray_collision_description(body_names=None)` 返回 companion `RayCollisionDescription`。它携带 `geom_names`、opaque 且 source-qualified 的 `geom_tags`、去重后的 `RayMeshDescription` asset、`geom_mesh_ids`，以及可选的无环 `body_parent_ids`。Parent ID 描述选中 public body 轴内的 ancestry；被省略的 native ancestor 会成为该轴的 root，因此需要完整 ancestor exclusion 的 caller 应选中相关 ancestor 链。Geom instance identity 是其 public index/name；mesh asset identity 独立于实例，多个 geom 可引用同一个 asset。两者仅在同一个 descriptor/materialization 生命周期内稳定；rebuild 会生成新 descriptor 并使旧 identity 失效。tag 没有 universal vocabulary：MuJoCo-family adapter 当前输出 `mujoco:geom_group:<n>` 作为 opaque、source-qualified tag，caller 通过 public descriptor 选择 tag，而不是使用 native 字段。所有数组均 detached 且写保护；native model、native mesh、Warp array、pointer 与 USD prims 均不跨过边界。

Portable mesh/hfield caller 通过 `collision=...` 将该 companion 传给 caster factory，并 materialize `collision.scene`；不需要 backend-specific builder。

## 生命周期

1. `create_ray_caster(name, num_envs=..., num_rays=...)` 在任何插件导入之前固定 batch 形状 `(num_envs, num_rays)`。
2. `create_ray_caster(..., collision=RayCollisionDescription)` 在冷路径绑定 detached companion，随后 `materialize(collision.scene)` 绑定不可变场景几何。所有 body 位姿初始为单位位姿。重复 materialize 会失败。
3. `update_pose(body_pos, body_quat, env_ids=None)` 写入世界系 body 位姿 `(rows, num_bodies, 3)` / `(rows, num_bodies, 4)`，作用于全部行或经过校验的选中行，要求 `supports_pose_sync`。
4. `trace(ray_origins, ray_directions, max_distance, env_ids=None, outputs=None, min_distance=0.0, ray_filter=None)` 投射固定的 ray 批并返回 `RayTraceResult`。
5. `close()` 幂等地释放资源；之后的查询必须失败。

tensor profile 额外要求显式绑定 pose 布局：

1. `bind_pose_layout(body_names)` 只解析一次不可变 body 顺序，且必须等于 `scene.body_names`。
2. `update_pose_tensor(RayPoseTensorView)` 传入调用方持有的 float32 Torch `(E, B, 3)` position 与 `(E, B, 4)` `wxyz` quaternion，并位于已声明设备。pose source 可以是 packed-strided view（例如 host-bridge packet 的列块），但最后一维 stride 必须为 1，且所有 stride 为正、不重叠；不支持的布局 fail closed。
3. `trace_tensor(ray_origins, ray_directions, max_distance, env_ids=None, outputs=None, min_distance=0.0, ray_filter=None)` 返回 persistent Torch view。

caster alias pose 存储而不是复制它；公开 quaternion 存储保持 `wxyz`，如需组件重排则在 kernel 内完成，并在下一次 tensor pose update 时替换 alias。场景替换是重建而不是变更：close，materialize 新 descriptor，然后重新 bind。

## Trace 语义

ray 使用世界系坐标且方向为单位向量。存在两种输入 profile：共享 profile 传入 `(num_rays, 3)` 数组并广播到每个选中行；逐环境 profile 传入 `(rows, num_rays, 3)` 数组，要求 `supports_per_env_rays`。`max_distance` 是正有限标量截断距离。最小结果为 `distance`（截断到 `max_distance`；未命中的 ray 精确报告 `max_distance`）和 `hit`。`env_ids` 选择经过校验的环境行子集；结果行按照选中顺序排列。

结果数组可能是实现持有的缓冲区视图，会被下一次 `trace` 调用复用；需要跨调用保留结果的调用方必须自行拷贝。实现不得依赖每次调用的无界内存分配。

tensor 路径接受同样的共享 `(num_rays, 3)` 与逐环境 `(rows, num_rays, 3)` profile，要求位于 caster 已声明设备上的 contiguous float32 Torch 输入，以及 contiguous 的一维 int64 `env_ids` tensor。其最小结果同样是截断后的 `distance` 与 `hit`；声明时可选用设备端输出提供 `hit_point`、`normal`、`geom_id` 与 `body_id`。caster 持有的结果 view 会被下一次 `trace_tensor` 复用；消费者必须在下一次 trace 前把它复制到 term 自己的 carrier。有限性与输出 ID 诊断延迟到 device-side validation 或显式 debug read，而不是强制热路径同步。

## 富查询语义

`min_distance` 是包含下界，`max_distance` 是排除上界：有效命中满足 `min_distance <= distance < max_distance`。下界必须严格小于上界。所有距离以及返回的 distance 都相对原始 ray origin。`min_distance == 0` 时零距离命中有效；miss（包括所有候选被 filter 排除或距离小于 `min_distance`）报告 `hit == False` 且精确为 `max_distance`。等距命中使用较小的 public geom index 作为确定性 tie-break。

`RayTraceFilter` 在单次调用内不可变，并对所有环境与 ray 相同。约束采用 AND 语义：`allowed_geom_ids` 与 `allowed_tags` 是 allowlist；`excluded_body_ids` 排除直接拥有 geom 的 body；`excluded_ancestor_body_ids` 在 owning body 或任意已选 ancestor 被排除时排除该 geom。ancestor exclusion 要求 `body_parent_ids`；未知 identity 或 tag fail closed。当前 contract 刻意不提供 literal bounded continuation：默认且唯一的语义是一次 nearest-valid-hit search。未来若需要逐 hop 兼容，必须作为独立 capability 与 contract 添加，而不是重载 `min_distance`。空 `RayTraceFilter()` 是显式 no-op，不要求 `supports_static_filter`；任一非 `None` 约束字段都会使请求成为 static filter。

Normal 在两种 transport 上均 optional。Mesh 命中使用 geometric face normal（不隐含 smooth/interpolated normal），primitive 命中使用对应 side/cap normal；caster 将 normal 变换到世界系，并在与 ray direction 点积为正时翻转。点积精确为零时返回有限 geometric normal，不翻转。Public mesh descriptor 拒绝退化三角形。`supports_host_normal` 与 `supports_tensor_normal` 分开声明，避免插件意外承诺未实现的 tensor 输出。在精确共享 mesh edge 上，等距 face 可能具有不同 geometric normal；Geom identity 与 distance 仍保持确定性，但除非 caster 声明更强的 face-index tie-break，normal parity 测试应排除这些测度为零的 edge ray。

## Capability 与 fail-closed 协商

`RayCasterCapabilities` 声明：

- `supports_pose_sync` —— materialize 之后的 `update_pose`。
- `supports_per_env_rays` —— 3-D 逐环境 ray profile。
- `supports_host_readback` —— `trace` 的 NumPy host 结果路径（当前一致性 helper 要求该能力）。
- `supports_tensor_pose_sync` / `supports_tensor_rays` —— Torch-native pose 与 trace 生命周期。
- `supports_device_output` —— caster 持有的 persistent Torch 结果缓冲。
- `torch_devices` —— 接受的 Torch 设备族或精确 CUDA index，使用 backend tensor contract 的拼写。
- `stream_event_ownership` —— 顺序所有权；当前 profile 只支持 `caller`。
- `supports_hit_point`、`supports_host_normal`、`supports_tensor_normal`、`supports_geom_id`、`supports_body_id` —— 可选结果字段，其中 normal 按 transport 拆分。
- `supports_static_filter` / `supports_min_distance` —— 静态 geom/body/tag filter 与包含下界。

可选输出通过 `RayTraceOutputs` 按调用请求。请求未声明的输出、在未声明 `supports_per_env_rays` 时使用逐环境 profile、或在未声明 `supports_pose_sync` 时调用 `update_pose`，都会抛出 `UnsupportedCapabilityError`；任何请求都不会被静默忽略或降级。命中点与法线为世界系，法线方向与 ray 方向相反，`geom_id`/`body_id` 按场景描述顺序索引（未命中时为 `-1`）。可选字段仅在 `hit` 为真时有意义。

capability 构造会做内部交叉校验：tensor pose sync 隐含 pose sync；tensor rays 隐含逐环境 rays；device output 隐含完整 tensor profile 与非空唯一的 Torch 设备列表；device output 要求 caller stream ownership。只声明 `cpu` 的 caster 必须拒绝 CUDA tensor，反之亦然，且不得把它们复制到支持的设备。

| Capability | Fake | `uniray` host 版本 | `uniray` tensor 版本 |
| --- | --- | --- | --- |
| Host NumPy pose/trace | 支持 | 支持 | 保留给工具、调试与一致性验证 |
| Tensor pose sync | CPU | 不支持 | 按实例声明的 CPU/CUDA |
| Tensor rays | CPU | 不支持 | 共享与逐环境 |
| Persistent Torch output | CPU | 不支持 | caster 持有的可复用 view |
| Torch 设备 | `cpu` | 未声明 | caster 接受的精确设备 |
| Stream ownership | caller | 不适用 | caller |
| Hit point / geom ID / body ID | host 与 tensor | host | tensor |
| Normal | host 与 tensor | host | tensor |
| Static filter / minimum distance | host 与 tensor | host 与 tensor | host 与 tensor |
| Mesh / hfield | public descriptor + 参考求交 | 取决于实现/profile | 当前 `uniray` profile 支持（hfield 在冷路径三角化为 mesh） |

## 参考实现与一致性

`FakeRayCaster` 以 NumPy 对基元与三角形 mesh 做解析求交，并提供 CPU Torch transport profile；mesh geom 要求 public `RayCollisionDescription` companion。它可以在没有 MuJoCo、Warp 或任何引擎 SDK 的环境中运行核心契约测试。`assert_ray_caster_conformance(caster)` 覆盖生命周期顺序、具有已知解析距离的规范地面场景、capability 声明的输出、声明时的富查询、两种 ray profile、选中行以及 fail-closed 拒绝——插件作者应对自己的实现运行该检查。

## 插件发现

`RAY_CASTER_SPECS` 声明插件身份（当前为 `uniray`）。插件包必须在顶层暴露 `create_ray_caster(num_envs=..., num_rays=..., **kwargs) -> RayCaster`；UniSim 工厂惰性导入它，校验返回值是 `RayCaster`，并在缺少包或入口点时给出可操作的 `OptionalDependencyError`。未知的 caster 名称抛出 `ValueError`。
