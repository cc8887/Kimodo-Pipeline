# 2026-10-11 独立 Kimodo 生产目录验证

## 交付范围

工具仓库为本独立目录，初始分支 `main`，本地 Git 管理，未配置远程或推送。包含当前作者 DAG、约束/派生/修复、ALS 脚本重定向、Root/Pelvis、曲线、最终 FBX 与回读、生产检查、Godot 消费适配器、Motion Viewer 和中文入门教程。安装采用源码 editable 模式；未验收 wheel 或离线安装包。

源工程为 Godot 消费项目，提取时 HEAD 为 `52c7ad514b090e22d2c248eb0c5dfffeea55f95d`。提取的是当时磁盘上的实际文件，包含已有未提交修改；逐文件来源哈希与当时未提交清单见 [extraction-source.json](extraction-source.json)。原生产文件未删除或覆盖，72 个原生产文件在忽略目录 `workspace/migration-source/` 保存字节一致副本。除了显式设置 `KIMODO_PIPELINE_HOME` 时使用新包的 `__init__.py` 适配入口，其余 71 个原生产文件与提取哈希一致。消费工程继续维护 Core/Import、场景和真实物理测试。

已建立参考快照 `workspace/als/`：665 文件、1,530,695,561 字节，原始导出字节/哈希检查通过；活动指针、compiled/import 缓存不进入快照。实际 local TOML、虚拟环境、模型、参考导出、制作包和运行日志全部忽略入 Git。

## 自动检查

| 检查 | 结果 | 当前机器证据 |
|---|---|---|
| 生产 Python 回归 | 255/255 | `workspace/pipeline-tests-final-v2.log` |
| 新目录交付与路径专项 | 6/6，包含在 255 中 | `workspace/independent-tests-final.log` |
| Viewer Python 服务 | 4/4 | `workspace/viewer-tests-final.log` |
| Viewer 前端 | 9/9 | `workspace/viewer-node-tests.log` |
| 本地路径/参考/依赖检查 | 通过 | `workspace/doctor.json` |
| Python compileall | 通过 | `python -m compileall -q tools` |
| Git 暂存/行尾/载荷边界 | 通过；10 个 tracked JSON 与磁盘字节一致 | `workspace/git-index-report.json`、`workspace/git-diff-check-final.log` |
| 原生产文件与活动包核对 | 通过 | `workspace/extraction-preservation-report.json` |
| WSL 实际路径适配及 Windows 文件回读 | 通过 | `workspace/wsl-metadata-smoke/report.json` |

生产回归使用现有 Kimodo Windows Python 3.12 环境及原 `offline-python-deps` 中的 ufbx；新 `.venv-win` 仅安装 NumPy/SciPy/ufbx 和 editable 工具。首次在轻量环境运行全生产回归缺 Torch 等生成依赖，失败日志保留为 `workspace/pipeline-tests.log`。后续一次回归暴露迁移后测试夹具仍把活动指针当作原始参考；按正确消费者根修正夹具，保留原断言和门限，日志保留为 `workspace/pipeline-tests-v2.log`。

WSL 路径专项在真实 Ubuntu Python 中创建受控元数据，调用实际 `wslpath -w`，Windows Python 再验证适配后的作者文件/raw/motion 路径都可读取。UE `/ALS/...` 资产身份、原制作清单及 raw 字节保持。这个检查验证文件与进程边界，不是新的模型生成。

## 真实转换与 FBX

以当前活动包已有作者源进行明确诊断回放，没有重新 diffusion：Refactored `A_Als_Stand_Pose`、`A_Als_Walk_Forward` 与 V4 `ALS_N_Walk_F`。三个独立源完成纯脚本重定向、Root/Pelvis 与逐资产曲线、原地 runtime raw、最终 FBX；全物理键 ufbx 回读 3/3、目标步态 2/2 通过。

制作版本为 `workspace/als/assets/generated/als_nonprop_migration_replay_20261011/`，包含 811 个清单载荷和三个所有权回执。制作清单 SHA-256 为 `a660687f153fa22e36198e8be216f043ec5aa023a4d24d65b8bbe6a17ddce84a`。转换回执在 `workspace/runs/migration-replay-20261011/conversion/pipeline-result.json`；回放日志为 `workspace/replay-conversion.log`。该小包缺 180 个非道具身份，保持 `productionReady=false`，不能用于宣称完整资源交付。

## Godot 消费验证和拒绝结果

适配器复制到消费项目同名新版本目录，制作原包保持原字节。Godot Optimize 构建零错误/警告、资源扫描、真实 Godot 导入、P2B 和 recovery_movement/aim/overlay/ragdoll/stop/overlay_prop 六域绑定均通过。绑定后的实际消费清单 SHA-256 为 `3cb1c8c44afc57231a8c2df1c5d17fe4fa959e1b80c6463c1b3746c6ae4abb3d`，与制作清单分开记录。

初次场景诊断错误使用 `walk-forward` 子片段名而非注册矩阵场景，命令失败和日志保留在 `workspace/runs/migration-replay-20261011/runtime/`。随后从 WSL 实际调用 Windows runtime bridge，选择 `ordinary-default-entry`、`stance-60-single`、`directional-single`，三个场景全部通过，载荷和动画设置没有变化。

完整矩阵入口仍返回拒绝：场景摘要 `total=3`、`passed=3`、`failed=[]`，但 `completeMatrix=false`、`executedAllScenarios=false`、`diagnosticPartialPack=true`，方向质量报告为 `Directional coverage not selected`。最终 `runtime_failed` 保留；未放宽断言、补造覆盖或切换默认包。实际证据在 `workspace/runs/migration-replay-20261011/runtime-wsl-corrected/` 的 `runtime-result.json`、`e2e-matrix/summary.json` 和相应日志。

消费项目仍选择 `als_nonprop_crouch_slip_20261011_v2`，其包哈希与 HTTP 检查时一致。只读检查和诊断部署未改选活动包。

## Motion Viewer

独立 Viewer 运行在 `http://127.0.0.1:10016/`。页面、JS、阶段 API、链路 API 和全部 12 个 raw/FBX 预览 HTTP/哈希检查通过，16 个检查记录于 `workspace/viewer-http-report.json`。三个资源各有 retarget/root/runtime/fbx 四个目标阶段。回放没有生成新的作者候选，Viewer 保持 generated/master 阶段缺失，不借用历史同名动作拼成新批次。V4 与 Refactored 均登记最终 FBX。

## 未覆盖与安装资料

本次没有执行新的完整非道具生成、全 CPU diffusion、全量 Godot/Air/Mantle/原生蒙皮矩阵、视觉人工验收、长时间或性能验收。WSL 能访问 RTX 5080，真实 Windows bridge 和路径适配已测试，但没有新建完整 WSL Kimodo 环境、重下大模型并跑 fresh diffusion。

[入门教程](../入门教程.md) 的下载、权限与安装步骤已根据 NVIDIA Kimodo、Hugging Face、微软 WSL、PyTorch、Python、Godot/.NET 的官方页面核查。CPU 社区入口为 [localai-org/kimodo.cpp](https://github.com/localai-org/kimodo.cpp)：当前 compact SOMA 输出和约束支持不满足本管线 SOMA77/空间约束接口，未安装或宣称可替换现有后端。旧 Windows 作者回执仅在对应 Windows 后端续作；WSL 制作原包供 WSL 续作，历史跨系统归档仍开放。
