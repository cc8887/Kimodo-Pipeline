# 模块化普通移动配置

`tools/kimodo_pipeline/modular_locomotion_rules.json` 是 Sprint、Sprint_Acceleration、Walk_Backward 三个语义的可运行起点，继承 production_rules.json。它不代表这三个动作或全部普通移动已经通过生产验收。

在已配置 `pipeline.local.toml` 和原生 ALS 参考快照的生产仓库，使用已激活的 Kimodo 生产 Python 环境执行（消费侧 `.venv-win` 不含生成依赖）：

```powershell
python tools/produce_als_resources.py plan --scope locomotion --rules tools/kimodo_pipeline/modular_locomotion_rules.json --motion Sprint --motion Sprint_Acceleration --motion Walk_Backward --output workspace/runs/modular-plan
python tools/produce_als_resources.py inspect --scope locomotion --rules tools/kimodo_pipeline/modular_locomotion_rules.json --motion Sprint --motion Sprint_Acceleration --motion Walk_Backward
```

`plan` 保存完整 graph.json，`inspect` 的 selectedOrder 才是此次实际依赖闭包：三个原生参考、三个候选、三个有界修复和六个双版本交付，没有未使用的 Stand/Run 或 discovery 生成依赖。全图 summary 的资产总数不是此次交付数。

配置中的 `motionScope` 把默认选择限制到这三个语义；省略 `--motion` 也不会意外生成整个旧生产图。`--motion` 可以继续缩小该集合。

实际消耗 GPU 前，可将同一命令的 `plan` 改为 `preflight` 并指定新输出目录。它执行骨架／原生参考及真实约束构造，核验周期能否放入候选、骨盆与 Root 的关系、支撑可达性和稀疏预算；不执行 diffusion，不把一致的约束当成合格动作。

修改 `promptModules.style` 为 feminine、neutral 或 athletic，再执行 plan/inspect。增删风格模块使用 styles 下的模块列表；功能模块负责动作、方向、相位、站姿，风格模块负责上身、手臂和表现。默认 feminine 保留直背、平衡头部、大臂稳定贴身、小臂摆动及明显手部位移；蹲伏模块倾身条件仍独立，但本配置未选择蹲伏动作。不强制 locomotionPosture 或 idle 上身参考。提示词声明不能代替生成结果测量。

三个动作速度分别为 6、5、1.5 m/s，最后一个向后。候选实际时长固定 3 秒，不设置会覆盖时长的 candidateCycles。native_als + nativePhaseEvidence 只提供计划和约束依据；preserveReferencePelvisPath 与 feet_only 自动同步 Hips，support2、half-phase、两个完整约束周期保持上身可受风格提示词影响。两个周期也能放入后退 Walk 的 3 秒候选，不能在计划中写入实际上放不下的第三个完整周期。约束锚点修正最多 15 cm。bounded 修复最多实际尝试 24 次、四个周期候选；保持原 20°/15 cm/5 cm 修正预算和严格 aftergate，失败不会交付为成功。每个交付合同保留 targetGaitPolicy，后续目标骨架/相位检查仍须实际执行。

正式执行把 plan 改成 run，并使用新的 output 和 pack-output 目录；需要保留其余动作来源时显式传 `--base-pack <已有包>`。可按需提供 `--upstream-receipts <回执>`：当前复用检查会验证提示词等生成输入的身份，修改风格后不会借用旧提示词的候选。不要使用 `--authoring-only` 作为完整交付：普通 run 继续重定向、Root/Pelvis 分解、曲线、FBX 回读、Godot 导入和相关运行门禁。base-pack 的历史结果不替代此次新资产验收。配置不保存机器路径或历史文本缓存。

计划回归（无需 GPU；CLI 测试需要配置的 ALS 合同快照）：

```powershell
python -m unittest tools.kimodo_pipeline.test_modular_locomotion_rules -v
```

此回归对三种风格分别执行真实 CLI plan/inspect，验证六个交付、依赖裁剪、速度方向、3 秒、原生相位约束和修复次数上限；它不执行生成或视觉验收。
