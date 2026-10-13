# Sprint 单周期裁剪与修复验证

已实现通用单周期裁剪／局部闭合，接入制作图；实际 Sprint 和 Sprint Acceleration 都得到 0.6 秒、19 个关键帧的单周期候选，首尾姿势一致。**两者尚未通过原源门禁，未交付最终可用 FBX，未切换默认包。** 此次不是解决全部快动作质量问题的证书。

## 选段与实际试验

源是 2026-10-13 `pelvis-controls` 批次的 `candidate:Sprint:pelvis:feminine` 和 `candidate:Sprint_Acceleration:pelvis:feminine`，原输入回执／所有载荷摘要均重新核验，没有执行新的 diffusion。原长候选各 90 帧；按 18 帧间隔只有 0–18、18–36、36–54、54–72 四个完整计划周期，72–90 缺第 90 帧。此前“四轮上限可能漏掉后段”的设想，在这两个输入中没有成立。

逐段普通修复共 16 次，另有 32 次接触／接缝变体、32 次有界局部闭合诊断及 8 次直接原段裁剪。每项都从原输入重做，不累积失败修改；没有将这一诊断枚举改为生产默认无限搜索。前两段可在原预算内构造候选；后两段的下肢端点差约 Sprint 6.89／15.46 cm、Acceleration 11.20／25.45 cm，既有约束修复前提继续拒绝，不能认为后段必然更好。

最终图选择原始第 18–36 帧（0.6–1.2 秒），使用现有接触修复和姿势闭合；不再强行更改已锁脚支撑帧的旋转切线。局部切线匹配的失败和预算证据单独保留，未放宽门限。

## 输出与风格

| 动作 | 关键帧／时长 | 首尾姿势／FK 位置差 | 累计修复角度／踝／Root 峰值 | 最终手部位移，左／右 |
| --- | --- | --- | --- | --- |
| Sprint | 19／0.6 s | 0／0 | 16.07°／1.93 cm／0.86 cm | 20.19／20.18 cm |
| Sprint Acceleration | 19／0.6 s | 0／0 | 13.99°／3.73 cm／0 cm | 20.21／20.25 cm |

修复预算不包含有意替换的上身风格；上身使用原固定大臂／小臂摆动 profile 并独立验收。先在长候选上构造小臂相位再裁剪，Acceleration 右手只有约 4.59 cm 位移，未达原 8 cm 要求。最终在单周期时间域重新制作上身，双脚／Root 字节保持，大臂相对胸部变化约 `1e-14°`，双动作姿态检查通过。证据为 `loop-posture-audit.json`（保留早期不足）和 `final-loop-summary.json`（最终重建）。

这表明选一段后需要重新处理该段的周期相关通道，不能只删除帧。以上是源骨架测量，未执行人工风格验收或目标蒙皮验证。

## 保留的失败与整链边界

制作图三次尝试均调用原 `run_batch` 和 `complete_run`，目标是两语义的四个 V4／Refactored 身份。最终版 `delivery-contact-crop-v3/completion.json` 明确 `pipeline_complete_rejected`、`failedPhase=source-quality`、成功交付 0。

- Sprint 仍缺独立检测到的完整交替接触周期；近地速度／接触判定含此前已证实的代理量问题，不能简单称为明显滑步，也不能将没有接触事件称作零滑步。
- Acceleration 有重复播放周期，但仍失败 `seamVelocityMps`、`seamAngularVelocityRps`。姿势闭合后的离散 FK 速度差约 3.02 m/s、角速度差约 6.05 rad/s；原门限分别为 0.75 m/s、6 rad/s，未改变。
- 重复一个修复 Loop 是边界播放测试，不是新增独立生成周期的质量证据。

无任何通过的生成交付时，既有整链入口明确阻断目标转换、Root/Pelvis、曲线、最终 FBX、Godot 导入和运行；没有用原版回退运行制造通过。原始 NPZ／原版 FBX、默认包指针和其他任务资产保持。

## Viewer 与证据

最终预览批次 **`single-loop-crop-20261013-v2`**：`生成候选` 为复用输入＋上身处理的 90 帧，`周期处理／母动作` 为最终 19 帧原地单周期。每个母动作 `gatePass=false`，资格明确“源门禁仍拒绝”。旧预览 `single-loop-crop-20261013-v1` 保留，未覆盖；当前最新批次指针未因手动预览登记而切换。

证据根目录 `workspace/runs/single-loop-crop-20261013-v1/`：

- `all-cycle-results.json`、`contact-variant-results.json`、`closure-results.json`、`raw-crop-results.json`：逐轮与有界变体。
- `production/`、`delivery-contact-crop-v2/`、`delivery-contact-crop-v3/`：不同尝试独立的源失败、选段／接触／累计预算和完整入口结果；不覆盖旧运行。
- `viewer-preview-v2.json`、`final-loop-summary.json`：最后预览身份和上身要求。
- `cli-smoke/`：可复用 CLI 真输入执行，明确源拒绝退出，输入未改写。
- `loop-segment-regression.log`：10 项控制测试通过，包括行进逆变换、inclusive 终点、保留中段、切线、预算、真实滑动负控、原门禁正控，以及失败阻断后继节点。

最终相关回归 **84/84** 通过（`focused-regression-v2.log`），扩展 Pipeline 回归 **495/495** 通过（`pipeline-regression.log`）；后加的两个控制例也纳入最终相关回归和单周期10项日志。扩展回归执行期间其他任务仍有既有变更，不能把这个计数归为本任务新增测试。受控测试中的伪造 Godot 阶段不代表本批真实 Godot 运行。限定改动 `git diff --check` 通过；默认指针仍是 `als_nonprop_crouch_stop_final_20261013_v5`，未提交或推送。下一步是依据明确的接触／插值证据处理快动作作者门禁与周期内轨迹，再沿既有整链交付；单周期化已经完成，最终资源验收仍未关闭。
