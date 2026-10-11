# 文档入口

- [从零入门教程](入门教程.md)：下载 Kimodo、申请 Llama 权限、Windows/WSL 环境、CPU 地址、生产命令、prompt 和 Viewer。
- [独立目录契约](architecture/standalone-workspace.md)：参考快照、制作包、Godot 部署副本、WSL 路径与验收边界。
- [本次迁移验证](verification/2026-10-11-independent-workspace.md)：源码来源、测试、FBX、真实导入/场景、失败与未覆盖项。
- [生产算法说明](../tools/kimodo_pipeline/README.md)：DAG、约束、派生与原质量门禁。
- [逐资产合同](architecture/kimodo-als-asset-contracts.json)：当前迁移的机器可读资源合同。
- [来源与许可边界](architecture/source-and-license-boundaries.md)：源工程的三份声明、独立目录中的资产范围和第三方依赖。
- [敏感信息修复验证](verification/2026-10-11-sensitivity-remediation.md)：路径清理、许可保留及本地初始历史清理的证据和边界。

其余 `architecture/kimodo-*.md` 来自 Godot 项目的设计文档快照。它们说明具体算法，文中旧时间点、历史包、消费侧 PowerShell 命令及历史验证链接仍属于原 Godot 项目；历史验证记录没有全量复制到此仓库。安装与日常操作按本仓库入门教程，目录边界按独立目录契约，本次迁移事实按本次验证记录。

算法文档中的 `python` 指已经激活并安装本工具的 Kimodo 环境。生产入口从当前 local TOML 读取后端和 Python 路径；消费侧 `scripts/verify-*.py` 仍须在 Godot 消费项目运行。个人机器路径只填写被 Git 忽略的 local 配置。
