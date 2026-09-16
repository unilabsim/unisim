# M2 IsaacGym 映射场景 worker

[English](../en/m2-isaacgym.md)

本文记录 [#108](https://github.com/unilabsim/unisim/issues/108) 工作包 C 的原生 worker 切片。共享 host adapter 尚未选择此路径。原生测试构造已协商的 payload，通过帧协议 IPC 和共享内存调用真实隔离 worker；它们不代表公开 `create_backend()` 多实体入口 或整个 M2 roadmap 已完成。

## 支持的 profile 与边界

- 每个实体或固定变体使用一个自包含 MJCF 源。原生 importer 接收显式 root mode 和 可安全导入的 XML；物化仍由 host 冷路径负责。
- 支持含 hinge/slide 的固定/浮动 articulation、浮动刚体、固定物体和 kinematic rigid 可视镜像。此 profile 不声明支持 ball joint。
- 构造期固定的任意 assignment（含 `[1, 1, 0, 1, 0]`）为每个消费实体从 K 个已加载 asset 中选一个；镜像必须使用相同 assignment。
- position drive 每个受控 joint 对应一个声明 actuator。被动关节使用 `DOF_MODE_NONE`，drive gains/effort 为零，且不占 action 列。非零源 joint passive damping 在独立验证前明确拒绝。
- 最多 30 个实体使用不同的物理碰撞 filter bit。物理实体之间允许碰撞；可视 actor 共享所有物理 filter bit，因此不与它们碰撞。各物理实体内部 self-collision 关闭； 这是显式 profile 近似，不代表通用 MJCF 碰撞语义等价。
- 不隐式创建地面；场景必须声明物理 ground/table。通用接触对查询、运行时 DR 与 wrench API 不属于本切片。
- 带关节的镜像源需由 host 烘焙为 rigid visual；worker 不自行删除关节或推断镜像 几何。完整场景离线 playback 和删除临时 legacy worker 分支仍待集成完成。

MuJoCo canonical XML 可能将源 `<position>` actuator 改写为 `<general>`。Gym importer 遇到不支持的 actuator tag 可能无限循环；worker 在调用原生 asset loader 前拒绝这些 tag。host staging 应删除导入文件的 actuator 元素，单独提供显式 drive records。 限位关节必须显式声明 `limited="true"` 与编译器解析后的 range；原生 `hasLimits/lower/upper` 需读回审计。原生质量、COM 与惯量必须匹配编译器记录；前期实测发现仅使用原始 `geom mass` 时， importer 会忽略该质量值。

## Wire 契约与原生绑定

INIT 携带 schema 1 的 `scene_layout`、有序 `scene_entities`、完整初始 `qpos/qvel/entity_root_state` 和重力。每个实体声明名称、kind、root mode、format、 collision/mirror、sources、assignment 及逐源 body/drive records。worker 使用公共 `scene_layout.py` 校验器，不导入 host 包。共享内存描述符必须精确匹配 `protocol.scene_slot_shapes()`，通过后才 attach segment。

worker 查询真实 actor/body/DoF indices，核对原生名称集合与 joint types，再映射为 公共 layout。创建后再次查询 actor 的 asset handle，反查实际 variant identity。 原生 mass、COM、inertia 和 drive properties 经读回与审计；META 的 `scene_entities_actual` 返回实际 assignment、source paths、actor IDs 和物理证据。 仅回传请求的 assignment 不能证明物化正确。

公共 root/body pose 描述 link origin，四元数使用 `wxyz`，不含环境 clone offset。 实测 Gym tensor API 已返回环境局部位置。原生线速度参考 COM，因此读回使用 `v_link = v_com - omega × R(q) * com_local`，写入使用逆关系。entity state 的 root 角速度为世界系；generalized root qvel 的角速度为 body 系。两种转换均考虑源 COM 偏移。

`RESET_ENTITIES` 携带选中行数/实体名及显式 root/qpos/qvel 写入 masks；所有校验在 native 提交前完成。actor root setter 和 DoF setter 使用查询得到的全局 actor IDs， 不能使用 env IDs。上一物理步以来待提交的写入累积为并集，否则重复 Gym indexed setter 可能覆盖较早的不相交 reset。仅成功推进物理后清空并集；native 提交失败将 worker 标为 faulted。

reset 后 root/joint state 新鲜；articulation 后代 body 的运动学状态可能直到 STEP 仍是上次求解状态，host 必须声明并执行该边界。刷新 body cache 不等于只做运动学 forward。

## 验证证据

开发期间在 **2026-09-17** 完成了以下小规模原生运行：

```sh
UNISIM_TEST_ISAACGYM_SCENE=1 uv run --extra mujoco pytest -q \
  tests/adapters/isaacgym/test_scene_native.py \
  --basetemp=/tmp/unisim-m2-gym-native-finalpass -x
```

全部三条验收测试在 **6.35 s** 内通过，顺序使用四个 worker 进程。 Runtime：IsaacGym Preview 4 `gym_38`、Python 3.8.20、Torch 2.4.1+cu121、PhysX GPU pipeline、NVIDIA GeForce RTX 4090、驱动 595.84。工作树基于 `df31d60bc7210fca551349eefeccd26ee65ff337`；这是开发工作树证据，不是最终 PR head 的验证声明。

| 检查 | 实测结果 |
| --- | --- |
| N=5、K=2、assignment `[1,1,0,1,0]` | 原生 object root 质量 `[3,3,1,3,1]` kg |
| 固定机器人 + 被动浮动 articulation + 桌面 + 镜像 | action 为一列；被动关节在 reset 后继续运动 |
| root pose/velocity 与非零 COM | 90° 姿态及独立位置有限差分 oracle 通过；速度容差 0.02 m/s |
| STEP 前连续三次不相交 reset | 两个 object reset 和 mirror reset 均在物理步后保留；step 前其他行不变 |
| 镜像阻挡下落路径与移到旁边的对照 | 480 步，最大 pose 轨迹差 `0.0` |
| 仅 kinematic rigid，nq=nv=nu=0 | 选中 pose reset 与 STEP 通过 |

仓库包含显式启用的原生测试和 CPU 回归测试。原生测试在 pytest 临时目录写入 `evidence.json` 与 worker 日志；这些 `/tmp` 文件是会话产物，不是长期公共证据， 上表保留简要观测。最终集成必须重跑最终 head，归档带版本证据，覆盖公共 host 路径，并执行 [#108](https://github.com/unilabsim/unisim/issues/108) 关闭门禁。缺少 vendor runtime 时必须显式 skip，不能视为原生验收通过。
