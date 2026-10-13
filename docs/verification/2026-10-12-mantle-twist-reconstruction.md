# Mantle 八骨 twist 重建验证

生产端在 `offline_retarget` v2、Mantle 蒙皮修复、stage 和消费运行前接入 `als-segment-swing-twist-half-v1`，契约见[实体 twist 重建](../architecture/kimodo-twist-reconstruction.md)。消费者的 IK 增量与角色事务由 Godot 工程的纯 .NET Core 实现。

本轮复用已验收的 High／两侧 Low 作者源，只重定向六个 Mantle 身份，没有重新 diffusion。局部重出验证同一作者批次和全部输入哈希；保留其他 155 个目标输入。新候选 `als_nonprop_mantle_twist_20261012_v2` 已生成八骨动态旋转、Root/Pelvis、34 份 motion 曲线和 44 份 extracted Root 曲线，161 个最终 FBX 全实体骨／全键回读通过。六个 Mantle 的最终 raw 各仅八根 twist 旋转变化，其他 155 个交付 raw 语义与 70 个道具保持。

Python 39/39，包含轴向／swing 分解、镜像／参考轴、跨 ±180°、错误叶子与奇异姿势拒绝、恒定 twist 重签哈希仍拒绝，以及既有发布／FBX／有界手修复。首轮遗漏 Kimodo 搜索路径造成 Root 阶段失败，保留 v1 原目录；v2 在完整环境验证后恢复转换，另建候选。

本机来源和生产证据：

- [重出结果](../../workspace/runs/mantle-twist-20261012-v2/reexport-result.json)
- [作者输入完整性](../../workspace/runs/mantle-twist-20261012-v2/input-integrity-after.json)
- [155 个姿势及六个 Mantle 对比](../../workspace/runs/mantle-twist-20261012-v2/pose-comparison.json)
- [六个 Mantle 最终 FBX 再次独立回读](../../workspace/runs/mantle-twist-20261012-v2/fresh-mantle-fbx-readback.json)
- [受控生产测试](../../workspace/runs/mantle-twist-20261012-v2/production-tests.log)

消费端 Optimize、实际导入／P2B／六域绑定、Release Core 76/76 和 Import 专项 23/23 通过。双模型 Low／1.5 m／2 m、Root 开关、单并行的 24 个组合及两组 GPU 共 9,360 帧通过，120 张截图，单并行及 headless／渲染完整帧一致。角色故障回滚 4/4。1.5 m 定位帧女模型左前臂 twist 已由近 0° 变为 41.25°。

仍为诊断候选，默认未切换。女模型肘完全伸直与腕部摆转保留，关闭手锁的扩展负对照女模型 0/2，旧包同样失败；原人偶 2/2 检出。包还缺 22 个源身份，选中八个 E2E 全通过但完整方向未选，完整适配器仍返回 `runtime_failed`；Air、静态 Mantle、平台／中断各 11/11 通过。没有放宽原阈值或将局部通过提升为完整生产／人工／性能认证。

消费端逐骨、蒙皮、哈希、日志与剩余问题以[Godot 完整验证记录](../../../Godot-Locomotion/docs/verification/2026-10-12-mantle-twist-reconstruction.md)为准。
