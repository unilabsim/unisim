# 触觉传感器插件契约

[English](../en/tactile-sensors.md) | [中文](tactile-sensors.md)

UniSim 提供后端中立的触觉计算生命周期。触觉实现负责自己的材料模型、求解器、设备缓冲区和输出布局；UniSim 只负责脱离引擎的场景、安装位姿、运行位姿输入、生命周期和字段元数据。导入 `unisim` 不会导入 Unitac、Torch、CuPy 或 CUDA runtime。

## 场景与几何

`TactileSceneDescription` 复用公共 `RayCollisionDescription` 表示碰撞几何，并复用 `RaySiteFrameDescription` 表示传感器相对安装体的局部位姿。`object_body_names` 选择 body；可选的 `object_geom_names` 将感知范围精确限制到指定碰撞 geom，默认使用这些 body 下的全部 geom。`geometry_state` 可携带与碰撞 geom 顺序对齐的逐环境快照，来源为 `SimBackend.get_ray_geometry_state()`。可在冷路径调用 `SimBackend.get_ray_collision_description(body_names=...)`、`SimBackend.get_ray_geometry_state()` 和 `SimBackend.get_ray_site_frames(site_names)` 构造场景。插件可以拒绝自身无法表示的几何随机化。几何改变后需要构造新场景并调用 `rebuild()`；插件可以拒绝逐环境拓扑或几何差异。名称和数组均脱离引擎所有权。SDF、FEM 矩阵、marker 网格和图像标定等求解器专属数据不属于该描述。

## 生命周期

```python
from unisim.factory import create_tactile_sensor

sensor = create_tactile_sensor(
    tactile_type="hydroshear",
    num_envs=64,
    config=tactile_config,
)
sensor.materialize(scene)
sensor.update_pose(body_pos, body_quat)
sensor.advance()
observation = sensor.read()
sensor.reset([0, 3])
sensor.close()
```

UniSim 固定加载 `unitac` 包，`tactile_type="hydroshear"` 选择触觉实现。UniSim 原样传递 `config`，不解析实现专属配置项。

| 操作 | 语义 |
| --- | --- |
| `materialize(scene)` | 一次性绑定碰撞几何和传感器安装位姿。缓存的 body 位姿从零平移和单位 `wxyz` 旋转开始；第一次 `advance()` 前调用者必须通过 `update_pose()` 同步实际位姿。 |
| `update_pose(pos, quat, env_ids=None)` | 替换所选行的世界位姿；输入位置形状为 `(rows, bodies, 3)`，单位 `wxyz` 四元数形状为 `(rows, bodies, 4)`。不会推进历史状态。 |
| `advance()` | 根据最近缓存的位姿计算一次观测，并且只推进一次历史状态。即使位姿未变，重复调用仍会重复推进。 |
| `read()` | 读取最近观测，不改变状态。`materialize()` 后且第一次 `advance()` 或 `reset()` 前读取属于实现错误；`reset()` 后可在第一次 `advance()` 前读取清零值。 |
| `reset(env_ids=None)` | 清除全部或选中环境的历史状态和观测值，并保留缓存位姿。 |
| `rebuild(scene)` | 仅在声明 `supports_geometry_rebuild` 时重新绑定静态几何。成功后会清除历史、使旧输出视图失效，并要求在下次 `advance()` 前重新同步位姿。 |
| `close()` | 释放资源；可重复调用。关闭后其他操作失败。 |

`TactileSensorCapabilities` 声明接受的 NumPy/Torch/CuPy 输入 profile、输出 profile、选中环境支持、几何重建支持和 CUDA stream 所有权。未声明的 profile 会 fail closed。

## 观测

`sensor.read()` 返回字段名到数组或张量的映射。`sensor.output_specs` 为每个字段声明完整形状（包含环境维度）、dtype、单位、坐标系、维度名称、缓冲区所有者和复用生命周期。尺寸可变的维度可标记为 `None`。插件自行选择输出结构；契约不要求规则 marker 网格或固定通道集合。

## UniTac 边界

`create_tactile_sensor(tactile_type=..., num_envs=..., config=..., **kwargs)` 延迟导入 `unitac`，并将 `tactile_type` 和不透明的 `config` 原样传入。Unitac 提供 `create_tactile_sensor(tactile_type=..., num_envs=..., config=..., **kwargs) -> TactileSensor`。

基础包保持 NumPy-only。Unitac 在创建或物化所选触觉类型时加载自己的可选计算库。
