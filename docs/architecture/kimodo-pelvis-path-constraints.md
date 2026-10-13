# 保留原生 Pelvis 残差的可选条件模式

`gait_constraints.materialize_gait_constraints` 的 `preserveReferencePelvisPath: true` 是显式可选模式。旧规则默认保持原行为。它区分 Kimodo 的两种输入：

- `smooth_root_2d`：指定的平滑轨迹，二维 `[world X, world Z]`。
- `root_positions`：实际 Hips 位置，三维 `[world X, world Y, world Z]`；水平位置可以包含相对平滑轨迹的 Pelvis 残差。

本地 Kimodo `FullBodyConstraintSet` 与 `EndEffectorConstraintSet` 接收这两个独立量。只有缺省 `smooth_root_2d` 时才把实际 Hips XZ 当作替代；模型条件构造会用全局关节位置减去平滑 Root，形成包含 Hips 偏移的局部位置特征。不能将显式 `root_positions.XZ == smooth_root_2d` 作为所有姿态的必要条件。

## 有来源的恢复

启用必须同时提供经过 `constraint_phase` 重验的原生相位证据，并保持 `preserveReferenceHeading: true`。当前模式要求原生速度、方向、周期和 FPS 与请求一致，不支持自行旋转或缩放后的匿名参考。

算法再次读取不可变原生载荷，以原生 Pelvis 世界位置得到残差，并核对整条 SOMA 参考的实际 Hips XZ 与它一致。相位重采样时：

```text
actual_hips_xz(frame) = requested_smooth_root_xz(frame)
                      + verified_native_pelvis_residual(source_phase)
```

平滑路径起点仍是 `[0, 0]`，heading 仍由原有明确的 Hips 朝向约束提供；不会为了消除残差而移动路径起点，也不会从 Pelvis 的摆动速度猜角色朝向。保留的残差来自原生骨盆，不是新放宽的误差预算。

路径校验比较“实际 Hips XZ 减去已验证残差”与指定平滑轨迹，仍使用原 `1e-8` 精度；另记 `maximumPelvisResidualErrorMeters`，逐帧核对残差。支撑锚点、IK 可达性、脚的位置/方向误差、髋部下移上限和生成后作者/目标/运行门禁均不改变。

## 当前 Python 后端的同步打包

当前 Kimodo postprocess 有两个相关行为：不包含 Hips 的 EE 提取会临时把 Hips XZ 替换成 smooth Root；脚 mask 则只识别 `left-foot`、`right-foot` 等命名类型，通用 `end-effector` 即使列出脚也不会置这些 mask。

本模式使用明确适配器 `kimodo-python-named-feet-then-hips-v1`：

1. 保留命名的 `left-foot`、`right-foot` 条件，确保后端脚 mask 存在。
2. 在两脚与 fullbody 帧的唯一并集上添加 `end-effector`、`joint_names: ["Hips"]`，放在命名脚条件之后。
3. 每个相同 frame 的 fullbody、脚与 Hips 都读取同一份姿态。后端把 fullbody 排到最后时仍得到相同值。

`validate_pelvis_constraint_consistency` 在调用后端前解码全部姿态，拒绝同帧位置/旋转冲突、丢失 Hips 帧、脚后没有同步 Hips、Root 路径/heading/残差冲突及未知约束类型。它不会靠“后者覆盖前者”掩盖输入矛盾。显式传入其他 `pelvisConstraintBackend` 会拒绝；其他后端需要自己的已验证适配器。

所有稀疏类型分别按唯一 frame 计数，仍严格少于 20。Hips 并集单独计入预算；不能因为每只脚各自低于 20 就忽略并集超限。本模式也明确禁止额外手朝向锁定。

新模式中的 `maximumConstraintCycles=N` 只生成 N 个完整周期的支撑锚点。闭合边界 N 不再被误当作下一周期的起点。请求周期数必须确实放得进 clip；额外 clip 末帧若开启 `constrainClipEndpoint`，按其真实相位记录为末帧条件，不宣称它构成另一个完整周期。旧默认模式的既有行为保持。

## 已验证配置与边界

可用于 3 秒的 Sprint、Sprint Acceleration、Walk Backward 对照：

```json
{
  "seconds": 3.0,
  "fps": 30,
  "preserveReferencePelvisPath": true,
  "preserveReferenceHeading": true,
  "maximumConstraintCycles": 2,
  "fullBodyPhaseCycles": 2,
  "fullBodyPhaseFractions": [0.5],
  "supportAnchorCount": 2,
  "boundaryContextFrames": 0,
  "constrainClipEndpoint": true,
  "constrainHandOrientations": false,
  "poseConstraintMode": "lower_body"
}
```

仍须填写对应 `targetPeriodSeconds`、`rootPath` 并提供原生相位证据。这里不能把 Backward 的 1.1333 秒周期误算成 3 秒内容纳三个周期。

纯数值及真实 CPU 后端验证覆盖三动作、`full_body` / `feet_only` / `lower_body`、SOMA77 / SOMA30 共 18 项。调用实际 `load_constraints_lst` 和 `extract_input_motion_from_constraints`，以重新 FK 验证每个目标脚／全身位置，而非只检查 JSON。实际 Hips 提取误差为 0，平滑 Root 数据逐字保持；float32 脚 FK 误差最大约 `1.91e-6 m`，使用 PyTorch 标准 float32 读回比较，生产脚接触门限未修改。错误打包顺序的独立负控产生约 9.10 cm Hips 误差，前置一致性校验会拒绝它。

3 秒／2 周期／半周期条件的 Hips 并集为 Sprint 10 帧、Sprint Acceleration 和 Walk Backward 各 14 帧；fullbody 为 6 帧。完整参数、条件、回执和 CPU 验证位于 `workspace/runs/generic-locomotion-20261013/preserved-pelvis-backend-probe-v3/`。

早期 probe 的两次失败保留在相邻初版和 v2 目录：一次错误地给 loader 指定浮点 dtype 导致索引被转换；一次为自定 3 微米诊断精度过严，实际 float32 全关节差为 3.057 微米。最终 probe 使用生产相同的无 dtype loader 调用，按 PyTorch 标准浮点读回比较，并保留绝对误差；这些是诊断调用/精度边界，不能混同为生产质量门禁通过或失败。

这里验证的是条件与后端输入提取的语义一致性，没有执行模型扩散或 motion correction，更不构成最终资产发布资格。生成与完整转换、FBX 回读和消费者验证由正常生产入口继续执行。
