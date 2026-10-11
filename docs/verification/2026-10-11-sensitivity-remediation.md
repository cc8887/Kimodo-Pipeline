# 2026-10-11 敏感信息与许可修复

## 修复范围

已清理先前审计确认的 12 份文档/示例中的 44 行机器路径，包括 Viewer 启动示例中的真实 Windows 用户目录。模板采用仓库相对路径、Linux `~/` 和 dotnet 环境变量；Windows 生产示例从 local TOML 读取 Kimodo Python。WSL 教程用当前目录进入共享仓库，不固定盘符。此外清理了补丁说明中的两个共享盘路径。

三份根目录声明 `LICENSE.md`、`ASSET_LICENSE.md`、`THIRD_PARTY_NOTICES.md` 从源工程原样复制，逐份 SHA256 校验一致。没有替换署名或改写许可正文。独立资产目录映射见[来源与许可边界](../architecture/source-and-license-boundaries.md)。

## 本地历史处理

此仓库只有一个未发布的初始提交，无远程、标签或其他分支。本次按修复请求更新初始提交，使旧个人路径退出可达历史；过期旧 reflog 并清理不可达对象，再对全部 Git 对象复扫。备份先于改动：仓库外保存可校验的 Git bundle、完整 `.git` 副本和 130 份原 tracked 文件，172 个 Git 文件逐个校验 SHA256。

备份位置仅记录于被忽略的 `workspace/sensitivity-backup-location.txt`；备份包含修复前的敏感路径，供本地恢复，分发时使用清理后的仓库。原始审计证据保留在 `workspace/security-audit-20261011/`，本次复核另存 `workspace/security-remediation-20261011/`。Git 现有账号署名与 GitHub noreply 邮箱保留。

## 验证

| 检查 | 结果与证据 |
|---|---|
| Windows/真实 WSL 模板解析 | 通过；仓库相对路径、`~/`、环境变量以及不同 cwd 的一致性；`windows-template.json`、`wsl-template.json` |
| 教程与两个 README 命令语法 | 15 个 PowerShell、19 个 Bash 代码块通过；`document-commands.json` |
| 配置读取后的后端入口 | `inspect --scope run` 通过，选择 33 个节点，无外部输入；`inspect-run.json` |
| 交付/路径专项回归 | 6/6；`path-tests.log` |
| 当前 local 配置与参考快照 | `doctor --verify-reference` 通过；`doctor.json` |
| Viewer 服务 | 首页和 `/api/stages` 均 HTTP 200，阶段 JSON 可解析；`viewer-http.json` |
| 许可和原有 JSON 字节 | 三份声明与源文件一致；10 份 tracked JSON 与修复前备份一致；`preservation.json` |
| Git 与敏感内容复扫 | 当前文件及全部可达/不可达 Git 对象，检查凭据、个人目录、机器路径、忽略边界及旧对象残留；`result.json`、`acceptance.json` |

表中证据均相对于 `workspace/security-remediation-20261011/`。最终 Git 验收要求：凭据模式候选和个人目录均为 0；文档/示例机器路径为 0；旧初始提交对象不可读取；无不可达对象；原有 JSON 字节、实际 local 配置不变；无运行数据或私密文件进入 Git。原 Windows 路径测试的三条合成输入与前端 FBX `P:` 属性解析正则保留，它们不是机器路径。

这是本地模式扫描与路径/入口验证。忽略的模型权重、原版参考和生成资产、大文件日志、仓库外 HF 凭据存储未做完整内容扫描；已保留的私有历史备份不属于待分发仓库。没有运行新的模型生成、整链 Godot 场景矩阵或人工视觉验收；既有资产验收边界仍见[迁移验证记录](2026-10-11-independent-workspace.md)。
