# M2 IsaacSim 映射场景实现

[English](../en/m2-isaacsim.md) | [中文](m2-isaacsim.md)

本文记录 [M2 #108](https://github.com/unilabsim/unisim/issues/108) 中 [工作包 D #111](https://github.com/unilabsim/unisim/issues/111) 的 worker 实现与 限定范围的原生证据，不表示 D 或最终集成审计 F 已完成。公共 host 场景路径尚未 与此 worker 接通并通过验收。映射场景的 renderer 复用了现有辅助实现，但尚未 通过原生验收。

## 已实现的支持范围

`backend/isaacsim/scene_worker.py` 面向独立安装的 Isaac Sim 5.1.0 / IsaacLab 0.47.2、Python 3.11、CUDA 运行时。每个具名实体持有独立的 IsaacLab `Articulation` 或 `RigidObject` view。支持的请求使用独立 MJCF 源文件、标量 hinge/slide 关节、冻结的公共场景布局，以及每个受控关节一个执行器的位置控制。 被动关节保留状态列，不增加控制列。已实现固定与浮动 articulation、单物理 body 的浮动刚体、固定单 body 桌面，以及关闭碰撞的运动学单 body 镜像。不隐式增加地面。

实体 variants 使用 K 个转换后的 USD 原型，通过 `MultiUsdFileCfg` 确定性轮转 分配。各实体独立生成，镜像共享源身份，但姿态独立。当前范围要求同一实体各 variant 具有相同公共布局与驱动属性。不同源的质量和惯量会保留，并与原生运行时读回值核对。

预校验拒绝 URDF/USD/SuperDex 源声明、非轮转 assignment、ball 关节、运动学 articulation、同一关节多个控制输入、被动关节刚度、variant 间不同的驱动设置， 以及包含多个物理 body 的 rigid 实体。特别是，articulation 目标的冻结几何若仍有 多个 body，其镜像不在支持范围内。这些是明确的支持边界，不是对所有实体或资产 类型的支持证据。源中的被动关节阻尼等特征还需要 host 做源语义校验；worker 不会 猜测 wire 中未声明的语义。接触力槽位保持确定性的零值，不据此声明接触传感器支持。

## 冷路径构场与原生身份

INIT 在启动 Kit 前校验完整布局、声明、标量关节记录、assignment、维度和初始 数组。转换使用显式源惯性参数与私有临时 USD 目录。USD 修改负责声明中的碰撞和 root 行为，并在应用指定 IsaacLab 驱动配置前清除 importer 创建的驱动。 可选的 `initial_ctrl` 使用执行器维度 `(N, nu)`，在初始状态写入后设置原生控制 目标。keyframe 控制值独立于关节位置，不从关节位置重建。

固定 root 需要两项独立修复。导入的 world joint 必须锚定到实际克隆 root 的世界 变换，包括环境偏移与初始姿态。此外，`ArticulationRootAPI` 必须从刚体 root link 移到包含整个资产的 prim。否则，即使姿态看起来固定，PhysX 仍可能将其报告为受到 外部 joint 约束的浮动 articulation。worker 读取原生 `is_fixed_base`，而不是 根据姿态稳定推断 root 类型。对 rigid 实体，不需要的 importer world joint 会 设为 inactive。仅从编辑层删除 prim 可能重新暴露引用层中的同一 joint，并不足够。

原生 view prim 路径建立显式的“公共环境编号→view 行”映射；原生 body/joint 名称 另外建立列映射。读取、控制、reset 和元数据都使用这些映射，不假设创建顺序或 prim 字典序等于公共环境顺序。

每个转换后的源带有不可变 variant 标记。worker 从实际生成的每个 prim 读取标记， 将观察到的 assignment 与请求比较。另外，它从实际 PhysX view 读取质量、COM 和 惯量矩阵，按原生映射重排后，与独立编译得到的源记录比较。articulation root 模式、关节类型、父子拓扑和驱动增益也会校验。仅回显请求的 assignment 不算原生 身份的证据。

## 状态与 reset 行为

worker 发布 `(N, nq)`、`(N, nv)`、`(N, nu)` 和 `(N, E, 13)` 场景缓冲。 浮动 root 的广义角速度采用 body 坐标系；实体 root 速度采用世界坐标系，参考点为 root link 原点。读回时减去原生 clone 偏移，写入时加回。显式 IsaacLab link 速度写接口处理 COM 偏移，不以 COM 速度接口替代。

`RESET_ENTITIES` 先复制所选行与 masks，再执行校验。它拒绝重复或越界环境、未声明 实体的写入、非法 masks、非有限值、非法 root 四元数、违反 root 模式的写入，以及 广义状态与 root 状态不一致。所有校验均在第一次原生写入前完成。关节 reset 使用 当前原生状态保留未选中的位置或速度通道。仅 reset 受影响的实体和选中的原生环境行。 原生提交开始后发生异常，包括刷新失败，会将 worker 标为 faulted；共享 host 必须 拒绝继续使用。不声明支持原生回滚。

## 可复现的限定验收

常规 CPU 检查不导入 Isaac SDK，覆盖支持边界拒绝、mask/selector 校验、独立的 九十度旋转预期、原生环境行重排、未指定关节通道的保留，以及写入后刷新失败导致 的 faulted 状态：

```bash
uv run --no-sync pytest -q tests/adapters/isaacsim/test_mapped_scene.py
```

显式启用的原生测试通过真实共享内存协议启动 production worker。请按现有运行时 说明准备独立 SDK，并安装常规 MuJoCo 开发 extra，用作独立源 oracle。测试不下载 SDK：

```bash
UNISIM_TEST_ISAACSIM_SCENE=1 uv run --no-sync pytest -q \
  tests/adapters/isaacsim/test_scene_native.py \
  --basetemp=/tmp/unisim-isaacsim-scene-acceptance
```

测试顺序运行三个 N=2 场景：浮动受控 robot、固定被动 articulation、浮动被动 articulation。每个场景均包含两种质量的 rigid variant pool、固定桌面和视觉镜像。 断言覆盖实际实例身份与源属性、`nu` 不包含被动关节、重力、指定实体和环境的 reset 隔离、后续 step 的状态持久性、带 COM 偏移的非单位姿态、受控关节响应，以及镜像 远离与重合时物体轨迹一致。INIT 超时为 240 秒，命令超时为 60 秒，清理阶段会终止 无响应 worker。各用例在 pytest 临时目录记录 `command.json`、`init.json`、 `meta.json`、`result.json` 和 `stderr.log`。默认跳过，不新增常规 CI GPU 要求。

2026-09-17 实施时，等价的直接 worker 探针已在本机 Isaac Sim 5.1.0 / IsaacLab 0.47.2、CUDA 运行时通过。最后一次探针包含原生环境行映射和浮动被动 articulation；此前探针覆盖固定被动和浮动受控 articulation。仓库测试是这些探针 的持续维护版本，仍须在集成后的最终 head 上执行。这些检查独立构造测试 INIT payload，属于 worker 验收，不能证明公共 host 集成、renderer 支持、任意 assignment、 所有源格式、全部 rigid 拓扑，或 D/F 已完成。
