# ADR：面向感知的 Ray 几何与富查询

[English](../en/adr-ray-perception-queries.md) | [中文](adr-ray-perception-queries.md)

## 状态

已接受用于 2.x ray-query contract（#472）。

## 背景

#462 的 tensor-native ray lifecycle 已移除 host 绕行，但 portable perception caller 仍无法获得 mesh topology，也无法表达 nearest valid hit。Mesh 点和三角形位于 `uni_ray` 私有 `MjBatchCollision` 中，caller 因此必须保留 native model 或调用 backend-specific builder。真实传感器还需要初始化时确定的 geom/body/tag 排除、用于拒绝自命中的最小距离，以及可选 camera normal。逐 hop continuation 只是一种应用兼容行为，不是 nearest-valid-hit 查询的必要语义。

## 决策

保留 `RaySceneDescription` 作为紧凑的 structure-of-arrays pose/geom contract，并新增 detached `RayCollisionDescription` companion。该 companion 携带稳定 geom instance name、opaque semantic tag、去重后的 `RayMeshDescription` asset、geom-to-mesh mapping，以及可选无环 body ancestry。Geom instance identity 与 mesh asset identity 分离。两者仅在同一个 descriptor/materialization 生命周期内稳定；rebuild 会产生新 descriptor 并使旧 identity 失效。Public type 中不出现 native model、native mesh、Warp array、pointer、stream、event 或 USD prim。

Tag 是 opaque string，没有 universal vocabulary。Adapter 可以输出 source-qualified tag（MuJoCo-family adapter 输出 `mujoco:geom_group:<n>`），但 contract 不定义 universal tag 集合，caller 通过 public descriptor 发现 tag。`RayTraceFilter` 消费 public geom index、tag、direct owning-body ID 与 ancestor body ID。约束在单次调用内不可变、对所有 row/ray 相同，并采用 AND 语义。`excluded_body_ids` 表示直接拥有 geom 的 body；`excluded_ancestor_body_ids` 表示 owning body 或任意已选 ancestor。Parent ID 描述选中 public body 轴内的 ancestry；被省略的 native ancestor 成为该轴 root。缺失 ancestry、未知 identity、未知 tag 均 fail closed。

`trace` 与 `trace_tensor` 接受 `min_distance` 和 `ray_filter`。有效命中满足包含下界与排除上界：

```text
min_distance <= distance < max_distance
```

下界必须严格小于上界。所有距离均相对原始 ray origin。`min_distance == 0` 时零距离命中有效；miss 精确返回 `max_distance`；等距命中使用较小 geom index。默认实现是一次 nearest-valid-hit traversal；本 contract **不**加入 literal bounded continuation、epsilon origin advancement 或 hop exhaustion。若未来需要严格逐 hop 兼容，必须作为独立 capability 和 ADR 引入，而不是重载 `min_distance`。

Normal 保持 optional，并按 transport 拆分为 `supports_host_normal` 与 `supports_tensor_normal`。Mesh normal 是有限 geometric face normal，不隐含 smooth normal；primitive normal 对应相交 side/cap。caster 返回 world-space normal，并在 normal 与 ray direction 点积为正时翻转；点积精确为零时保留有限 geometric normal。Public mesh descriptor 拒绝零面积三角形，因此 normal 总有定义。精确共享 mesh edge 可能返回任一相邻 face normal；Distance 与 geom identity 仍保持确定性，但除非未来 contract 增加 face-index tie-break，normal parity 排除这些测度为零的 ray。

本阶段 sensor attachment 仍为 body-only。任务在应用代码中应用本地 sensor offset，并通过既有 public tensor pose contract 提供 body pose。Site/geom frame view 仍在范围外。Public terrain projection 由 #474 单独跟踪；本 ADR 只覆盖 collision geometry 与 ray query。

Host 与 tensor 实现必须在文档化浮点容差内保持 hit、distance、identity、filter 与 normal 语义一致。Tensor path 保持 caster-owned persistent output、caller stream ownership、无隐式 host readback、无 steady-state unbounded allocation。

## 后果

Portable caller 可以只通过 public descriptor materialize caster，不需要 `MjBatchCollision`、native model 或 backend-specific builder。Adapter 拥有冷路径投影，并在无法表达 runtime geometry、identity 或 ancestry 时 fail closed。应用继续拥有 grouped ray pattern、camera intrinsics、noise、history、scheduling 与其他 perception 逻辑。未来 bounded-continuation contract 可以在不改变 nearest-valid-hit 默认语义的情况下单独添加。

MuJoCo 与 MJWarp fixed-variant 场景不在首个 collision descriptor 中投影，因为 variant 可能改变逐环境几何；这些 adapter fail closed。Motrix 复用已审计的 ray source 选择，并继续拒绝 runtime-only geometry、attached fragment、逐行 mesh variant、异构 fixed-variant source，以及 source hfield/mesh geometry，直到 native introspection 或 terrain 跟进定义精确投影。

## 相关决策

- [Ray tensor lifecycle ADR](adr-ray-tensor-lifecycle.md)定义 pose ownership、caller stream 与 persistent tensor output。
- [Tensor lifecycle ADR](adr-tensor-lifecycle.md)定义 backend tensor execution 与 packed host-bridge plan。
- [Terrain projection roadmap](https://github.com/unilabsim/unisim/issues/474)单独跟踪 public hfield/terrain metadata。
