# 使用实际生成风格姿态的可选边界参考

`style_reference.compose_upper_body_reference(native_reference, generated_style_motion, rig, fraction=.5)` 返回 `(reference, receipt)`。它用于足部／下肢条件仍不足以控制周期边界时的明确备选，不把完整原生上身风格强加到所有提示词，也不替代已有 `feet_only`、`lower_body` 对照。

算法取实际生成 discovery 的一个上身姿态，以 Hips 的水平朝向对齐每帧原生参考，再把 Spine1 及其后代的目标全局旋转反解为局部旋转。包含脊柱、头颈、双臂、手和手指。原生 Root、Hips、所有下肢局部矩阵保持；执行后分别检查脚底和下肢 FK 不变。输入同时要求完整 SOMA 层级、可验证的全局姿态缓存以及一致的显式骨序／rig 摘要（若携带）。缺失链、不同 rig、陈旧 FK 缓存、无定义的水平 heading 均拒绝。

参考上身固定成这个生成姿态，目的仅是稀疏周期首尾锚点。约束配置应明确使用 `poseConstraintMode: full_body`、`fullBodyPhaseFractions: []`、`boundaryContextFrames: 0`，并关闭额外手朝向固定；周期中间仍交给文字和生成模型。若加入密集 fullbody 相位或手约束，可能意外抑制小臂摆动，不能把这种改动当作等价的对照。

例如：

```python
reference, composition_receipt = compose_upper_body_reference(
    native_reference, actual_discovery_motion, rig, fraction=.5,
)
phase_evidence = build_native_phase_evidence(
    immutable_index_path, native_source, reference, rig,
)
```

调用者必须绑定 discovery 的已执行生成节点身份、回执和文件哈希。该函数记录两个实际输入 motion 的内容摘要、rig 摘要、采样 frame/时间、上身骨清单、修改骨清单、保留骨清单及足部误差；单独一组数组的内容摘要不能证明它来自模型生成。

`constraint_phase.build_native_phase_evidence` 对合成后的参考重新执行原生载荷／目标门禁／toe 轨迹绑定。原生源审计文件保持，合成参考的实际审计另记；不得改写旧源审计。两个参考都只能用于条件，不能作为最终资产交付。

回执固定 `deliverable: false`、`productionReady: false`、`referenceAcceptedAsMaster: false`，`styleQualityStatus: not_evaluated`。生成文本描述了某种风格、以及取到了该候选的一个姿态，都不等于艺术风格已经合格。真正输出仍执行作者周期、目标骨架、最终 FBX 和消费者验证。

本轮数值兼容检查使用实际执行的三动作 × 三风格 discovery。九份参考的 Root/Hips/下肢数组及脚底／下肢 FK 均保持，误差为 0，独立 native phase 重验通过；这是参考构造验证，不是使用该参考重新生成后的成功率。证据位于 `workspace/runs/generic-locomotion-20261013/style-reference-probe/summary.json`。相应单测见 `test_style_reference.py`。
