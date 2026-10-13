# 从多周期候选制作单周期移动资产

长候选是选材，最终 Loop 通常只交付一个完整周期。本流程适用于速度／方向／站蹲合同明确的普通循环移动，不适用于起步、停止、跳跃的非循环阶段。`Sprint_Acceleration` 当前 ALS 合同是 0.6 秒循环，不是从静止开始的整段加速。

## 选段与修复

从实测同侧落脚事件到下一次同侧落脚事件寻找完整周期，或使用经过文件摘要验证的约束相位计划提出候选，再独立检查生成结果。不能只按相似姿势截取半步，也不能假定最后一轮有完整终点。优先比较周期中段接触、骨盆残差、下肢首尾、速度、上身风格及预计修复量；现有 `repair_selection.py` 支持多个周期的有界尝试，指定片段则使用下述 `loop_crop` 节点。

`loop_segment.crop_loop` 保留 inclusive 首尾关键帧，移除明确的线性平面行进并保存精确逆变换，保留骨盆周期残差和上下起伏。原地 Loop 的首尾 Root 可以相同；恢复世界行进后 Root 应前进一个步幅，不能把行进也强行拉回起点。

制作节点默认先保留未闭合裁剪，再做接触与最终时域的风格处理，最后调用 `loop_seam.repair_loop_seam`，然后执行独立作者门禁。原始接缝只作为诊断和修复输入，不要求它在修复前通过。`gait.py` 的验收阈值保持，直接使用旧 `gait_cycle` 的路径也保持；这里调整的是单周期制作链中的顺序。

最终修复默认固定末帧姿势和 Root 残差，令首帧等于它；可用 `seamAnchor: "first"` 固定首帧。两侧相邻键沿 SO(3) 共享切线，再将旋转／Root 的修正以 quintic smootherstep 衰减到未修改中段。优先保留非零切线；离散 FK 曲率仍超限时逐步减弱切线并尝试不同边界窗口，不提高阈值。只覆盖最后一帧会留下速度折角；把整段插值成一个姿势则会丢失步态，两者均不作为默认方案。

接缝同时检查原离散 FK／角速度和 SLERP／NLERP 的独立左右侧子帧 FK 速度。子帧测量用两侧各自的插值段，不用环形中心差分相等的恒等式审批。恢复明确的平面行进后，每个预算内的接缝通过候选还须经过完整支撑／穿地／近地位移门禁；如果全失败，只保留接缝已修复的诊断候选，`playbackGatePass: false`，制作节点仍失败，后续交付仍阻断。

接触修复沿用现有周期拟合、世界落脚锚点、固定骨长 IK 和自由脚端清地。边界 Easing 会改变腿部补偿速度，可能新增滑步或穿地，所以接缝平滑不能替代接触验收；若所有局部方案均失败，需返回接触／切线联合约束阶段，不能简单在最后补一次 IK 后跳过接缝复核。原角度／踝／Root 预算继续执行。

多阶段修正按同一帧和同一元素累计路径，不能将不同骨、不同帧的最大值相加当作真实累计修改。`periodic_repair` 可选输出 `correctionPaths`，供裁剪阶段映射和累计；默认关闭，不改变既有修复行为。

**先确定最终周期，再按其完整相位重建小臂动作。** 在三秒候选上只构造一轮小臂摆动，再截其中 0.6 秒，会截掉大部分手部位移。固定大臂／摆动小臂 profile 在裁剪后按最终腿部相位重建，并重新测量手部位移；蹲伏 profile 继续保留前倾。任何后续接触修复、风格修改或重采样都要复核首尾与支持。

## 可执行入口

### 接触与接缝联合修复

`jointContactSeam: true` 调用 `loop_contact_solver.solve_loop_contacts`。验证过的原生相位只提供修复意图，实际支撑、清地、近地速度和接缝均重新从输出 FK 测量。优先选择修复后实际内部左脚落地事件作为循环起点，排除播放启动边界的一帧检测伪影。

先完成小臂等风格通道，再平滑上身接缝。腿部不再直接随全身 Easing 衰减，而是用固定骨长求解同时满足世界脚尖／脚跟锚点、支撑速度、摆动清地及边界切线。默认固定投影后的端点；`jointOptimizeSharedAnchor` 可在预算内联合选择共同端点姿势，首尾共享同一组变量，保持严格相等。`jointOptimizePelvis` 可允许骨盆参与，并补偿其上身子链以保持修复前的上身世界朝向；最终风格仍独立验收。

修改路径按同一帧和骨映射累计。旋转与 Root 用欧氏球约束保留原 20°／0.15 m／0.05 m 限制；踝修改另计 FK 路径。优化权重 `jointSeamWeight`、`jointSupportWeight` 和 `jointNearGroundWeight` 只能在 1–10 内调整误差优先级，不能改变验收阈值；`jointMaximumEvaluations` 为主求解的 1–500 次上限。软 FK 罚项仍超踝预算时，以更强罚项和 0.1 mm 预留再求解至多 100 次，仍按原 0.15 m 判定，不能四舍五入或容忍微小超限。未收敛不等于拒绝或通过，最终独立测量决定；任一预算或检查失败仍拒绝交付。

验收通过后保存的动作不可再被通用上身处理覆盖。后续 `gait_cycle` 使用 `postureAlreadyFinal: true`，核对来源的通过回执，并复核上身风格。源接受不等于目标接受：若目标自身垂直低带不能建立 IK 窗口，可使用已核实输入摘要、相同时间／速度的源 FK 接触相位，再以目标垂直摆动／低带交叉检查修复意图。目标已通过时不启用后备修复；修后仍执行原目标几何／滑步／支撑门禁，且保持 8 cm 踝／5 cm 骨盆修复预算。源的 `FootPlanted` 或声明接触曲线不能代替 FK 相位。

### 普通生产入口

目标前脚掌支撑仍失败时，先保留失败候选，再从同一原始目标试算有限个修复：脚掌朝向稳定权重为 0.05／0.1／0.2／0.25／1，支撑窗口只允许增加至多一个收尾帧，且该帧的实际目标脚尖高度须落在原窗口低带上沿加 1 cm 内。新增帧是修复意图，不是改写验收接触或放宽质量门禁；每个候选仍独立执行原目标检查。脚掌与脚趾朝向总修正不超过 20°，踝仍限制为 8 cm。若仅下移骨盆会超限，求双腿固定骨长可达球交集内的最小共同骨盆位移，禁止抬高，三维位移总长度仍不超过原 5 cm；不能只限制垂直分量而增加平移预算。首尾一致、骨长保持。所有通过候选中，选双脚近地速度相对原门限的最大比例最小者，为真实蒙皮复验保留余量；没有新增通过阈值。全部失败则保留诊断，不交付。回执记录未扩展窗口、候选列表、采用窗口和实际修改量。

骨架代理量通过后必须再看真实模型蒙皮。本轮 V5 的目标代理量通过，游戏 Sprint 蒙皮近地速度仍为 27.65 cm/s，超过原 25 cm/s；该失败保留，不能被骨架通过覆盖。最终候选须经过原版同场景对照和普通单线程／并行／渲染三种消费方式的独立检测。源、目标、FBX 和蒙皮门禁分别绑定各自实际文件与输出。

增量合成还须处理派生顺序。Overlay 表重新组装站／蹲下肢后，如果声明了 twist 重建且最终检查拒绝，重建八个实体叶子 twist 的旋转，再执行原最终检查；不能把组装前的通过回执直接套到组装后输出。大臂、脚部及其他主骨骼不会由这个步骤重写。保留源按其实际原作者回执核对，不使用合并字典中可能发生同名碰撞的其他分支记录。

规则文件的 `loopCrops` 将指定语义接入原生产 DAG，无需编写独立 Runner。各段必须等于原资产周期键数，速度从原 `gait_cycle` 合同继承，不能覆盖；接触参考必须有独立相位证据。`source` 应选择本次生成的最终风格输入，避免把已被原始接缝拒绝的旧修复节点当作裁剪前提。

`tools/kimodo_pipeline/joint_loop_rules.json` 是可执行起点，继承模块化提示词、固定大臂／小臂摆动和蹲伏前倾 profile。它不是任意风格、动作或 seed 的通过承诺。本轮实际复用候选与该新生产配置的意图不同，不能无视摘要强制复用。

```powershell
python tools/produce_als_resources.py plan --scope locomotion `
  --rules tools/kimodo_pipeline/joint_loop_rules.json --motion Sprint_Acceleration `
  --output workspace/runs/my-loop-plan

python tools/produce_als_resources.py run --scope locomotion `
  --rules tools/kimodo_pipeline/joint_loop_rules.json --motion Sprint_Acceleration `
  --base-pack <current-resource-pack.json> --keep-going `
  --runtime-case ordinary-default-entry --runtime-case directional-single `
  --runtime-case directional-parallel --runtime-case rendered-directional `
  --output workspace/runs/my-loop-delivery
```

更换风格先改 `promptModules`，再核对上身 profile 是否仍符合新意图。仅改修复参数时，可提供与本图意图一致的 `--upstream-receipts`，只重算制作和验收。输出保留 `joint-loop-solver.json`、源／目标相位、失败原因及最终文件回读。完整包状态与单资产相关场景通过分别记录。

诊断一个指定片段，不覆盖输入或已有输出：

```powershell
python tools/crop_motion_loop.py --motion <motion.npz> --rig <rig.json> `
  --start-frame 18 --end-frame 36 --speed 6 --pose-only `
  --output workspace/runs/my-single-loop-v1
```

帧号从零开始，末帧包含在内。示例在 30 FPS 下是 18 个时间间隔、19 个关键帧、0.6 秒。`--pose-only` 仅做旧端点均值闭合并诊断，不执行新的最终修复。省略它时执行固定末帧的接缝修复，可用 `--seam-anchor first` 固定首帧。输出 `single-loop.npz`、恢复行进的 `playback.npz` 和完整 `loop-crop.json`。源检查拒绝时退出码为 2；此诊断命令不执行 FBX 或 Godot，也不是发布入口。

制作图使用通用 `loop_crop` 节点，继续接原 `gait_cycle` 和资产 deliver 节点，最后调用原完整 Pipeline：

```json
{
  "id": "crop:Sprint",
  "op": "loop_crop",
  "deps": ["rig:soma", "candidate:Sprint", "reference:Sprint:native-als", "motion:Idle"],
  "params": {
    "source": "candidate:Sprint",
    "startFrame": 18,
    "endFrameInclusive": 36,
    "seamFrames": 3,
    "expectedSpeedMps": 6,
    "contactReference": "reference:Sprint:native-als",
    "harmonics": 3,
    "finalSeamRepair": true,
    "seamAnchor": "last"
  }
}
```

这是节点结构示例，不是独立完整图。`contactReference` 可省略；提供时必须有绑定的 `native-phase-evidence.json`，选段与参考周期起点对齐，先修复该原始片段的接触，再裁一轮。可提供既有 `postureProfile` 和 `postureValidationIdle`，在最终时域重建上身；它们同时须声明图依赖。新的最终接缝修复默认开启，`finalSeamRepair: false` 仅用于重现旧裁剪／闭合诊断。`pre-seam-loop.npz`、`loop-seam-repair.json` 和失败播放都保留，本轮未覆盖历史证据。

成功的 Loop 交给 `gait_cycle` 进行合同帧数整理、行进逆变换和曲线来源验证，再依次执行目标重定向／足部检查、Root/Pelvis、逐资产曲线、最终 FBX 回读、Godot 导入和相关运行验证。源阶段无成功资产时，完整入口明确记录 `source-quality` 阻断；不得构造只有原版回退的包来证明新 Loop 通过。

## 验收与快速复用

输出检查至少包含首尾姿势、Root 高度与残差、关节角速度、FK 速度、循环接触、支撑漂移、穿地、实际插值及动作风格。19 帧中的首尾表示同一相位，播放和统计不能把末端关键帧额外当成一个时间间隔。重复播放四次可测边界支持，却不能证明原始生成具有四个独立稳定周期；回执明确保留这个区别。

已有生成回执继续按实际文本、条件和依赖摘要复用，换提示词不能直接沿用旧生成。只修改选段、边界窗口或修复参数时复用未变的生成，独立重算后处理和验收。每个候选保留原始来源、选段、实际修改路径、成功／失败检查和 Viewer 阶段；只把完成整链验收的资产计入可用资源。

原单周期结果见[单周期试验](../verification/2026-10-13-single-loop-crop.md)，最终接缝修复与接触边界见[本轮验证](../verification/2026-10-13-anchored-loop-seam.md)；源滑步代理量的误归因另见[门禁复核](../verification/2026-10-13-sprint-slip-gate-audit.md)。
