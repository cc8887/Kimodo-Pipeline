# ALS 实体 twist 重建

SOMA77 没有对应 ALS 的 twist 关节。`offline_retarget.convert` 的 v2 策略按目标骨架重建八个实体分支，避免未映射辅助骨始终保持参考局部旋转。算法只修改 twist 的局部旋转，保留主骨、手掌／足、指骨、参考偏移、缩放和原时间轴；它不复制某段原版动画的 twist 轨道。

`als-segment-swing-twist-half-v1` 使用不可变参考姿势定义骨段轴。上臂／大腿的近端 twist 抵消父主骨轴向旋转的一半，前臂／小腿的远端 twist 分担手／足相对参考的轴向旋转的一半。每根 twist 必须是主骨的直接叶子分支，骨段终点是同父兄弟骨。参考数组可以包含尾部虚拟骨，实体重建只使用 `rawBoneNames`／`rawParents` 对应前缀。

对主驱动骨局部旋转 `R` 与参考 `R0`，计算 `D = R R0⁻¹`。把四元数向量部投影到骨段轴，保留标量部，得到轴向 twist；swing 不参与分配。近端轴在主骨父空间取 `R0 × endpointReferencePosition`，输出绕主骨局部骨段轴旋转；远端直接使用终点参考偏移轴。左右侧使用各自真实轴，不能硬编码同向 X。

逐键轴角先 unwrap，再乘分配比例；输出四元数保持相邻键同一半球。否则原主骨越过 ±180° 时，半比例 twist 会突然翻转约 180°。精确 180° 的纯 swing 无法定义轴向扭转，本策略明确拒绝，而非以任意旋转认证通过。0 轴向运动允许 0 twist，不要求所有动作出现非零 twist。

接触及周期 IK 会改变主骨。stage 对带 `twistReconstructionPolicy` 的新转换重新烘焙最终 twist；Mantle 手部修复的每次蒙皮候选也使用同一重建，防止用旧辅助骨形变计算新的指节／掌部避障。Root/Pelvis 分解后按最终 raw 重算八骨一致性，记录到 `target-twist/<sourceHash>.json` 与 `targetTwistChecks`；所有报告和最终 raw 都进入载荷摘要。消费运行前重新计算一致性，恒定 twist 加上新的文件哈希仍会被拒绝。最终 FBX 独立回读仍覆盖全部实体骨、全部键。

Godot/Core 的 `AlsTwistBoneReconstruction` 对运行 IK 使用增量：比较 IK 前后驱动骨的轴向旋转，取最近的等价角差，再将一半差值加到输入中已经烘焙的 twist。这样保留新生成或原版已有的扭转分布与分支连续性。驱动手部 IK 和换模接触 IK 在原角色候选内依次调用；没有新增时钟、历史状态或提交链路，无 IK 扭转变化时保留原值。叶子旋转不改变手端点／骨长。

局部重出可用 `offline_pipeline --retain-bundle <oldBundle> --retarget-source <exactSource>`，每个源均先验证作者输出、身份和合同；保留项另验证 raw／FBX／NPZ／BVH 哈希。保留 bundle 必须绑定同一作者批次，所有输出使用新包目录，后续 Root、曲线、FBX 与 Godot 验证照常执行。它复用已验收生成成果，不伪称新 diffusion；保留项不会自动获得新 twist 认证。

半比例是当前蒙皮策略，不是原版动画等价认证。辅助骨正确驱动也不认证关节限位、肘部屈曲、肩／身体姿势、蒙皮体积、自交或美术风格。后续可依据目标蒙皮增加分配 profile；现有原生、骨长、接触与穿模门槛保持。
