# Mocap 姿态与几何重置随机化

[English](../en/mocap-reset-contract.md) | [中文](mocap-reset-contract.md)

状态：已接受。该决策扩展现有后端边界；不引入第二套环境生命周期、张量协议或引擎实现。

## 决策

`SimBackend.bind_mocap_pose(body_name)` 在冷路径解析一个固定 mocap body，并返回 `unisim.backend.base` 的 `BackendMocapPoseBinding`。不可变绑定包含 `backend_type`、`body_name`、`num_envs`、形状为 `(7,)` 的分离只读 `default_pose`，以及后端拥有的回调。其公共方法是：

- `read() -> np.ndarray`：分离的 `(num_envs, 7)` 世界姿态。
- `write(env_ids, poses) -> None`：唯一整数环境 ID 与浮点 `(len(env_ids), 7)` 姿态。布局为世界坐标 xyz 后接单位 wxyz 四元数。空选择是经过校验的 no-op。

适配器应用姿态写入，前推物理以刷新派生状态，并在返回前刷新传感器与状态缓存。广义位置、速度以及未选中的 mocap 姿态被保留。消费者不得保留原生 model/data 句柄，也不得在 reset 或 step 代码中解析名称。未知名称、非 mocap body、非法形状、dtype 或 ID、非有限值，以及非单位四元数都会显式失败。其他后端的默认实现在绑定时抛出 `NotImplementedError`；不能从类名或私有方法推断能力。

`set_state` 保留既有重置语义：被选中的环境将 mocap 姿态重置为模型默认值。同时修改广义状态与 mocap 姿态的消费者先提交 `set_state`，再执行绑定 mocap 姿态写入。这些是有顺序的操作，不是跨后端原子事务。这保持了既有 `set_state` 签名稳定，并让环境 owner 负责组合重置的顺序与预校验。既有绑定在重置和步进后保持有效。

## 重置随机化表

`ResetRandomizationPayload` 为 `R` 个选中行添加以下稠密表：

| 字段或能力项 | 形状 | 含义 |
| --- | --- | --- |
| `geom_size` | `(R, ngeom, 3)` | MuJoCo 基本几何尺寸 |
| `geom_solref` | `(R, ngeom, 2)` | 接触参考参数 |
| `geom_solimp` | `(R, ngeom, 5)` | 接触阻抗参数 |
| `dof_damping` | `(R, nv)` | 非负关节阻尼 |
| `dof_frictionloss` | `(R, nv)` | 非负关节摩擦损耗 |

`unisim.dr.types` 中的常量是 `RESET_TERM_GEOM_SIZE`、`RESET_TERM_GEOM_SOLREF`、`RESET_TERM_GEOM_SOLIMP`、`RESET_TERM_DOF_DAMPING` 和 `RESET_TERM_DOF_FRICTIONLOSS`。`get_geom_sizes`、`get_geom_solref`、`get_geom_solimp`、`get_dof_damping` 与 `get_dof_frictionloss` 为冷路径消费者绑定暴露分离默认表；它们不是逐世界当前模型视图。模型随机化在仅重置状态的 reset 后持久存在。

MJWarp 在改变状态或上传模型数据前校验新表。Solref 接受两个正的时间常数/阻尼比值，或两个非正的直接格式值。Solimp 要求阻抗端点在 `[0,1]` 内、宽度为正、中点在 `(0,1)` 内，且幂至少为 1。几何尺寸写入要求基本几何半径或范围为正；未使用分量可以为零。所有表值都必须能表示为有限 float32。

支持球体、胶囊体、椭球体、圆柱和长方体缩放。适配器一次性分类几何，使用缓存索引组从新尺寸派生 `geom_rbound` 与 `geom_aabb`，并在前推前原地上传全部三个字段。消费者不能独立提供不一致的包围盒。稠密载荷可以包含未变化的 mesh、plane、hfield 或 SDF 列；试图缩放这些不支持类型会失败。几何缩放不会隐式改变质量或惯性；需要时消费者显式请求这些字段。固定地址逐世界扩展发生在 CUDA graph 捕获之前，因此后续写入保留已捕获指针。

## 宿主几何 reset 契约

`ResetRandomizationPayload` 还定义以下宿主侧稠密几何表。`R` 是按调用方顺序排列的选中环境行数；`ngeom` 遵循冻结的公共 geom 顺序。提交的表为每个公共 geom 列提供最终绝对值。`None` 表示省略该项，不恢复默认值也不修改其他项；仅写入尺寸不会清除已选的 mesh variant。运行时能力由各 adapter 的 `supported_reset_terms` 独立协商：存在载荷字段不等于原生支持。这些宿主字段不属于 `TensorResetRandomizationPayload`。

| 字段 | 类型和形状 | 含义 |
| --- | --- | --- |
| `geom_size`（既有） | float `(R, ngeom, 3)` | 几何尺寸；支持的尺寸规则由 adapter 决定 |
| `geom_active` | bool `(R, ngeom)` | 逐行几何启用状态 |
| `geom_pos` | float `(R, ngeom, 3)` | 相对于所属 body 的几何位置 |
| `geom_quat` | float `(R, ngeom, 4)` | 相对于所属 body 的几何姿态，wxyz 顺序 |
| `geom_shape` | 分类字符串 `(R, ngeom)` | `sphere`、`capsule`、`cylinder`、`box`、`ellipsoid` 或 `mesh`，不是引擎原生枚举 |
| `geom_mesh_variant` | int `(R, ngeom)` | 对应 geom 已注册 mesh variant set 中从零开始的序号，不是固定模型 variant 索引 |

五个新项在 `unisim.dr.types` 中对应 `RESET_TERM_GEOM_ACTIVE`、`RESET_TERM_GEOM_POS`、`RESET_TERM_GEOM_QUAT`、`RESET_TERM_GEOM_SHAPE` 和 `RESET_TERM_GEOM_MESH_VARIANT`。Adapter 声明能力前必须校验维度、dtype、geom/variant 组合及原生行为。reset 时的几何选择不改变构建时的 `FixedVariantPlan` assignment。Portable Motrix 在来源几何支持时声明 `geom_size`，并在来源几何与已注册 mesh set 允许时声明 `geom_shape`/`geom_mesh_variant`。固定依赖的 MotrixSim Core `0.10.2.dev126386` 还提供 `geom_active` 和局部 `geom_pos`/`geom_quat` 原生 override。Adapter 在修改前校验选中行，通过原生逐实例 override 写入，不在运行时逐次回读校验；原生 setter 报错会使后端进入 faulted 状态，不支持的场景与 tensor reset 快速失败。切换 shape 或尺寸不会隐式修改质量或惯量。上文既有的 MJWarp 缩放声明仍限于所述基本几何，不意味着 shape 切换或 mesh 缩放。

## 证据与限制

`tests/contract/test_reset_capabilities.py` 覆盖严格绑定、默认后端行为和载荷项过滤。`tests/adapters/mjwarp/test_required_capabilities.py` 覆盖选中世界的 mocap 平移/旋转与重置、几何和接触效果、接触参数力变化、阻尼与摩擦运动效果、非法请求，以及对照官方 MuJoCo 编译的基本几何包围盒。数值测试需要 3.11 MJWarp extra 和 CUDA；被跳过的运行时测试不是支持声明的数值证据。

本变更不添加相机或 raycast 地形、后子步传感器钩子、任意场景变体、力矩扰动或 mesh 缩放。既有原生命名传感器绑定仍是接触、执行器力和 site 速度读取机制。
