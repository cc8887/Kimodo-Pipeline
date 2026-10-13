# 文档入口

- [单周期联合接触修复与游戏接入](verification/2026-10-13-joint-loop-game-integration.md)：最终风格后共享端点／接触联合求解、原预算、目标脚掌候选排序、真实蒙皮失败保留与增量合成；[可执行制作方法](architecture/kimodo-single-loop-crop.md)。
- [多周期候选制作单周期 Loop](architecture/kimodo-single-loop-crop.md)：指定选段、行进逆变换、接触与累计预算、最终风格后固定端点 Easing 修复和完整验收；[原裁剪试验](verification/2026-10-13-single-loop-crop.md)、[最终接缝修复与支撑失败边界](verification/2026-10-13-anchored-loop-seam.md)。
- [Sprint 滑步门禁复核](verification/2026-10-13-sprint-slip-gate-audit.md)：近地速度误归因、支撑边界与脚跟滚动、240 Hz 插值、真实漂移负控和 Viewer 原版回退来源。
- [可替换风格的移动制作方法](architecture/kimodo-style-resilient-locomotion.md)：门禁复核、模块化提示词、原生相位／骨盆约束、有界修复与增量交付；[可运行配置](architecture/modular-locomotion-rules.md)与[本轮证据及仍失败的边界](verification/2026-10-13-style-resilient-locomotion.md)。
- [普通移动上身风格](architecture/kimodo-upright-locomotion.md)：站立接近 Idle、固定大臂／小臂摆动、蹲姿保留前倾、来源绑定与原门禁；[本轮制作与整链结果](verification/2026-10-12-locomotion-upper-body.md)。

- [从零入门教程](入门教程.md)：下载 Kimodo、申请 Llama 权限、Windows/WSL 环境、CPU 地址、生产命令、prompt 和 Viewer。
- [独立目录契约](architecture/standalone-workspace.md)：参考快照、制作包、Godot 部署副本、WSL 路径与验收边界。
- [本次迁移验证](verification/2026-10-11-independent-workspace.md)：源码来源、测试、FBX、真实导入/场景、失败与未覆盖项。
- [生产算法说明](../tools/kimodo_pipeline/README.md)：DAG、约束、派生与原质量门禁。
- [逐资产合同](architecture/kimodo-als-asset-contracts.json)：当前迁移的机器可读资源合同。
- [来源与许可边界](architecture/source-and-license-boundaries.md)：Pipeline 非商业许可、原工程声明保留、独立资产范围和第三方依赖。
- [非商业许可验证](verification/2026-10-11-noncommercial-license.md)：官方许可正文、原声明、包许可元数据与敏感内容检查。
- [敏感信息修复验证](verification/2026-10-11-sensitivity-remediation.md)：路径清理、许可保留及本地初始历史清理的证据和边界。

其余 `architecture/kimodo-*.md` 来自 Godot 项目的设计文档快照。它们说明具体算法，文中旧时间点、历史包、消费侧 PowerShell 命令及历史验证链接仍属于原 Godot 项目；历史验证记录没有全量复制到此仓库。安装与日常操作按本仓库入门教程，目录边界按独立目录契约，本次迁移事实按本次验证记录。

算法文档中的 `python` 指已经激活并安装本工具的 Kimodo 环境。生产入口从当前 local TOML 读取后端和 Python 路径；消费侧 `scripts/verify-*.py` 仍须在 Godot 消费项目运行。个人机器路径只填写被 Git 忽略的 local 配置。
