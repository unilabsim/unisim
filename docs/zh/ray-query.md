# Ray-Query 插件契约

[English](../en/ray-query.md) | [中文](ray-query.md)

UniSim 定义了 backend-neutral 的 ray-query 插件契约，使 ray caster 实现（例如独立分发的 `uni_ray` 包）可以在不暴露引擎私有类型的前提下，为任务代码提供批量 ray 查询。第一版只覆盖最小闭环：生命周期、固定形状的批量 trace、capability 声明和 fail-closed 的输出协商。

## 公开接口

以下名称均可从 `unisim` 包根安全地惰性导出，实现位于 `unisim.ray_query`：

- `RayCaster` —— 抽象的插件生命周期：`materialize(scene)` → `update_pose(...)` → `trace(...)` → `close()`。
- `RaySceneDescription` —— 冷路径消费的 backend-neutral 结构 of arrays（SoA）场景描述。
- `RayCasterCapabilities` —— 细粒度的 capability 声明。
- `RayTraceOutputs` / `RayTraceResult` —— 单次调用的输出请求与结果容器。
- `RayGeomType` —— 几何基元类型（`plane`、`sphere`、`box`、`cylinder`、`capsule`、`ellipsoid`，以及预留的 `mesh`）。
- `require_ray_trace_outputs(...)` —— fail-closed 的输出协商辅助函数。
- `RAY_CASTER_SPECS` / `ray_caster_spec(...)` —— 已声明的插件清单。
- `create_ray_caster(...)` —— 惰性工厂分发。
- `FakeRayCaster` —— 纯 NumPy 的解析参考实现。
- `assert_ray_caster_conformance(...)` —— 供插件作者复用的一致性检查。

## 类型边界

公开接口只接受 NumPy。`mujoco.MjModel`/`mujoco.MjData`、Warp kernel 或数组、CUDA 指针以及任何其他后端私有类型都不得出现在签名或结果中。真实的 adapter 在自己一侧把原生场景与位姿状态翻译为 `RaySceneDescription` 和位姿数组。导入 `unisim` 不会导入任何引擎 SDK；插件包由 `create_ray_caster` 惰性导入，缺少包或入口点时以 `OptionalDependencyError` fail closed。

## 场景描述

`RaySceneDescription` 为每个 geom 携带一条 SoA 记录：`geom_types`、`geom_sizes` `(n, 3)`、`geom_local_pos` `(n, 3)`、`geom_local_quat` `(n, 4)`（单位 `wxyz` 四元数）和 `geom_body_ids` `(n,)`，外加显式的 `num_bodies`。geom 在其所属 body 的局部坐标系中表达。尺寸遵循 MuJoCo 风格的基元约定：sphere 为 `(radius, 0, 0)`；box 为半尺寸 `(x, y, z)`；cylinder 与 capsule 为 `(radius, half_length, 0)`，轴向为局部 `+z`；ellipsoid 为半轴 `(x, y, z)`；plane 是无限的局部 `z = 0` 平面，法线为 `+z`，忽略其尺寸。所有数组在构造时校验、拷贝并写保护；非法输入 fail closed。

## 生命周期

1. `create_ray_caster(name, num_envs=..., num_rays=...)` 在任何插件导入之前固定 batch 形状 `(num_envs, num_rays)`。
2. `materialize(scene)` 在冷路径绑定一次不可变的场景几何；所有 body 位姿初始为单位位姿。重复 materialize 会失败。
3. `update_pose(body_pos, body_quat, env_ids=None)` 写入世界系 body 位姿 `(rows, num_bodies, 3)` / `(rows, num_bodies, 4)`，作用于全部行或经过校验的选中行，要求 `supports_pose_sync`。
4. `trace(ray_origins, ray_directions, max_distance, env_ids=None, outputs=None)` 投射固定的 ray 批并返回 `RayTraceResult`。
5. `close()` 幂等地释放资源；之后的查询必须失败。

## Trace 语义

ray 使用世界系坐标且方向为单位向量。存在两种输入 profile：共享 profile 传入 `(num_rays, 3)` 数组并广播到每个选中行；逐环境 profile 传入 `(rows, num_rays, 3)` 数组，要求 `supports_per_env_rays`。`max_distance` 是正有限标量截断距离。最小结果为 `distance`（截断到 `max_distance`；未命中的 ray 精确报告 `max_distance`）和 `hit`。`env_ids` 选择经过校验的环境行子集；结果行按照选中顺序排列。

结果数组可能是实现持有的缓冲区视图，会被下一次 `trace` 调用复用；需要跨调用保留结果的调用方必须自行拷贝。实现不得依赖每次调用的无界内存分配。

## Capability 与 fail-closed 协商

`RayCasterCapabilities` 声明：

- `supports_pose_sync` —— materialize 之后的 `update_pose`。
- `supports_per_env_rays` —— 3-D 逐环境 ray profile。
- `supports_host_readback` —— `trace` 的 NumPy host 结果路径（当前一致性 helper 要求该能力）。
- `supports_device_output` —— 为未来的设备端结果路径预留；本版本中不启用任何功能。
- `supports_hit_point`、`supports_normal`、`supports_geom_id`、`supports_body_id` —— 可选的 `RayTraceResult` 字段。

可选输出通过 `RayTraceOutputs` 按调用请求。请求未声明的输出、在未声明 `supports_per_env_rays` 时使用逐环境 profile、或在未声明 `supports_pose_sync` 时调用 `update_pose`，都会抛出 `UnsupportedCapabilityError`；任何请求都不会被静默忽略或降级。命中点与法线为世界系，法线方向与 ray 方向相反，`geom_id`/`body_id` 按场景描述顺序索引（未命中时为 `-1`）。可选字段仅在 `hit` 为真时有意义。

## 参考实现与一致性

`FakeRayCaster` 以纯 NumPy 实现该契约，对六种基元类型做解析 ray 求交；mesh geom 在 materialize 时 fail closed 拒绝。它可以在没有 MuJoCo、Warp 或任何引擎 SDK 的环境中运行契约测试。`assert_ray_caster_conformance(caster)` 覆盖生命周期顺序、具有已知解析距离的规范地面场景、capability 声明的输出、两种 ray profile、选中行以及 fail-closed 拒绝——插件作者应对自己的实现运行该检查。

## 插件发现

`RAY_CASTER_SPECS` 声明插件身份（当前为 `uni_ray`）。插件包必须在顶层暴露 `create_ray_caster(num_envs=..., num_rays=..., **kwargs) -> RayCaster`；UniSim 工厂惰性导入它，校验返回值是 `RayCaster`，并在缺少包或入口点时给出可操作的 `OptionalDependencyError`。未知的 caster 名称抛出 `ValueError`。
