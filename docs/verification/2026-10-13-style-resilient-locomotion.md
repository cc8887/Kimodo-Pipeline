# 可替换风格的移动制作：2026-10-13 验证

后续更正：本记录的源“滑动失败”指旧近地速度代理量超限，不能直接证明支撑期明显滑步。Sprint／Acceleration 修复例已发现边界差分与离地脚跟混入，详见[滑步门禁复核](2026-10-13-sprint-slip-gate-audit.md)。历史拒绝结果保留；另有插值穿地和接缝等独立问题，未因此放行资产。

本轮按“门禁合理性 → 模块化提示词 → Kimodo 约束 → 后处理 → 整链交付”执行，证据根目录为 `workspace/runs/generic-locomotion-20261013/`。稳定方法见[可替换风格制作方法](../architecture/kimodo-style-resilient-locomotion.md)，可执行配置见[模块化移动配置](../architecture/modular-locomotion-rules.md)。没有提交、推送或切换默认动画包。

## 已确定的门禁与故障边界

- 原生目标正控 22/22 通过。Sprint 新增两个版本的目标路由；Sprint_Acceleration 为 0.6 秒循环，原先排除周期修复的分类已纠正。错误速度和抬高动作负控继续拒绝。
- 原 2 cm 脚底支持阈值合理。原人偶原版／候选通过，女模型原先缺样本来自实际鞋底厚度与踝骨位置的差异约 3.47 cm。通用接触标记重定向后，两模型各 28/28 通过；采样阈值未改变。
- 源接缝的 0.75 m/s、6 rad/s 标准比原生快动作更严格。高阶差分和真实插值推导均不足以证明可安全换掉它，原门禁保留。失败参考可凭独立的原生目标／脚尖 FK／文件摘要证据供应条件，不能成为交付。
- 5 cm 髋部修正预算曾拒绝由错误过渡制造的 11.75 cm 降髋需求。修复过渡改为消退边界修正，原失败例降到约 1.27 cm；该例仍未通过其他源条件。
- 清地时原来整体抬脚会破坏脚尖支撑，现按固定脚尖／脚跟旋转处理自由端。真实例恢复左右各 15 帧支撑，脚尖漂移约 `5.5e-15 m`，但未据此隐藏接缝／滑动失败。
- 跨不同帧、不同骨的阶段峰值简单相加会误拒预算；改为同一元素修改路径累计，不能用净位移抵消。seam1 三例预算通过但源接缝仍失败；seam2 的约 28.5°／24.2–25.1 cm 实际累计修改继续拒绝。

详细证据在 `gate-audit/`、`budget-audit-production/`、`postprocess-pinned-sole-regression/`，接缝推导另见[诊断记录](2026-10-13-seam-gate-diagnosis.md)。

## 提示词与真实生成

固定模型 `Kimodo-SOMA-RP-v1.1`、SOMA、30 FPS、100 steps、seed `2026101217`，对 Sprint／Sprint_Acceleration／Walk_Backward 比较旧文本、中性、当前柔和风格和运动风格。全部主对照为 3 秒，生成设置与实际参数保存在每个节点回执；稀疏输入、约束数量及失败按阶段独立保存。

| 实验 | 实际 diffusion 数 | 结论 |
| --- | ---: | --- |
| 原文本与三个模块化风格，只有 Root 路径 | 12 | 全部源门禁失败；部分文本出现周期，但文案本身不能保证接触 |
| 脚部相位约束，保留上身自由 | 9 | 后退 Walk 后处理后三风格通过；Sprint 类仍失败 |
| 脚部与 Hips 条件 | 9 | 新增 Hips 本身没有解决快动作问题 |
| 增加摆动相位与约束 CFG | 6 | 有接触改善，仍未通过全部接缝／预算 |
| 本风格 discovery 上身用于周期边界 | 6 | 参考保持原生下肢且不可交付；仍未获得合格 Sprint |
| 显式保留骨盆相对 smooth Root 的残差 | 9 | 后退 Walk 三风格通过；Sprint 类仍未全部通过 |

共 51 次实际 diffusion；复用、约束参考构造和离线修复不计新生成。没有完成多 seed 成功率评估，不能称任意风格或 seed 都稳定成功。

## 约束与快速迭代能力

`prompt_modules.py` 已接入实际 Graph，生成设置保存有效模块和实际文本。方向／相位冲突、固定大臂数值 profile 与摆臂文本冲突均拒绝。复用检查覆盖实际 prompt、模型／seed／步数／CFG、相关条件依赖及动作摘要；父 prompt 改变不能被后代旧回执绕过。

发现并修复 Hips XZ 被强制等于 smooth Root 的问题。原生骨盆残差峰值约 Sprint 2.57 cm、Acceleration 9.93 cm、Backward 4.12 cm；丢掉该残差会将原生加速支撑区脚速从约 0.025 m/s 放大到 1.356 m/s。新模式按同一量核对路径与骨盆残差，保留命名脚 mask，并拒绝同帧冲突、错误顺序和稀疏并集超预算。

真实 CPU Kimodo SOMA77／SOMA30 loader 与 postprocess 输入提取共 18 项通过；Hips 误差 0，脚 FK float32 最大误差约 `1.91e-6 m`。这验证输入打包，不能替代实际生成检查。证据为 `preserved-pelvis-backend-probe-v3/`。

可提交配置限定三个语义，`--motion` 进一步缩小范围。已提供 `preflight` 命令：实际构造参考与约束，报告一致性并明确 `generationExecuted=false`。本轮公共配置三个动作预检均通过，`public-preflight-v1/preflight.json` 保留完整报告；普通 `run` 仍执行完整转换与消费验证。

有界修复选择最多 24 次，仅对实际接触／接缝失败追加对应措施。后退 Walk 可在 1–3 次尝试内成功；相同输入重放 NPZ 字节和报告一致。保留骨盆的新三风格后退源，拟合修改约 10.45–16.14°、踝修正约 1.09–1.55 cm、髋下降 0，均在原预算内。

## 消费端与交付

显示和支撑预测使用同一组真实蒙皮接触标记；实际显示骨架跟随新的 `PresentationSkeleton` 接口。GPU 捕获绑定请求／提交帧，必要时提交后显式绘制，后台窗口不再造成漏图。消费侧 80 项 Core、相关 Python 检查、双模型方向 28/28、所选 10 场景、6,720 帧单／并行／渲染精确一致、112 张方向和 36 张释放图通过。完整过程见消费项目 `docs/verification/2026-10-13-locomotion-runtime-gate-repair.md`。

交付脚本 `deliver_controls.py` 将每种风格按六个目标身份重放原作者门禁，失败分支继续保留，成功后退身份进入完整 Pipeline。feminine 分支显式应用原有固定大臂／小臂摆动 profile，保持本轮最初的动作要求；另外两种风格不套用该 profile。

最终逐包结果由 `delivery-results.json`、`final-delivery-verification.json` 和各 `delivery-*-v1/completion.json` 记录：

| 风格候选 | 新完成身份／选择身份 | 最终 FBX | 相关运行场景 | 方向足底／单并行渲染一致 | 最终载荷摘要 |
| --- | ---: | ---: | ---: | --- | ---: |
| `als_nonprop_modular_locomotion_20261013_feminine_v1` | 2／6 | 179／179 | 10／10 | 通过 | 2,246／2,246 |
| `als_nonprop_modular_locomotion_20261013_neutral_v1` | 2／6 | 2／2 | 10／10 | 通过 | 807／807 |
| `als_nonprop_modular_locomotion_20261013_athletic_v1` | 2／6 | 2／2 | 10／10 | 通过 | 807／807 |

三个候选的后退 Walk 两版本都通过源、目标和实际显示检查，均无失败目标 gait。feminine 合并保留 V8 其他来源；neutral／athletic 是仅两个生成身份的对照包，其余动作使用明确的原版回退。不能把三个版本的 FBX 数量相加后当成 183 个独立身份齐全。最终交付重放复用本轮已执行且摘要核验的生成回执，没有增加 diffusion。

所有包整体仍为 `pipeline_complete_rejected`：Sprint／加速四个身份仍未通过源门禁，所选相关矩阵也不等于完整生产矩阵。feminine 保留动作的 Air 和两类 Mantle 回归通过；扩展跳跃仍失败，女模型左侧长滞空落地约穿入 4.01 cm，原人偶也保留落地失败。这些不属于后退 Walk 成功的覆盖范围。

最终完整 Pipeline 回归 **401/401** 通过；约束公共预检三个语义通过且不执行生成，限定改动 `git diff --check` 通过。默认包与模型配置摘要仍与本轮开始前一致。没有完成多 seed 成功率、全部动作／性能／长时间或人工风格验收。本轮新解法未证明 Sprint／加速可交付，不能把原生参考、对照输入或曾失败的修复 NPZ 放入成功清单。

进一步的全局下移和联合 pelvis／腿／支撑接缝探针仍未通过全部门禁，保留在 `pelvis-velocity-probe/`、`joint-seam-probe/`、`joint-seam-probe-v2/`，均为非交付诊断。局部 C1 优化不是当前生产默认策略。
