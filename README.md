# Kimodo ALS Pipeline

[简体中文](README.md) | [English](README.en.md)

将 NVIDIA Kimodo 生成的动作整理为 ALS 动画资源，并通过独立的制作检查及 Godot 消费端验证完成交付。

> **许可提示：** 本仓库自有的集成代码、脚本、测试、规则及相关文档采用 PolyForm Noncommercial 1.0.0。原有 MIT 代码、模型、第三方资产与导出数据有各自的授权范围；请先阅读 [NOTICE](NOTICE)、[ASSET_LICENSE.md](ASSET_LICENSE.md) 和[许可边界说明](docs/architecture/source-and-license-boundaries.md)。

## 项目概览

Kimodo 负责生成源动作；本 Pipeline 管理动作依赖、ALS 资源制作与验证。生产流程为：

| 阶段 | 内容 |
| --- | --- |
| 生成与派生 | 按动作依赖图生成源动作、姿态和派生资源 |
| 目标制作 | ALS 重定向、有界修复、Root/Pelvis 分解和逐资产曲线 |
| 文件验收 | 回读最终 FBX，核对生产回执及资源清单 |
| 消费端验证 | 在 Godot 项目导入资源、构建并运行相关检查 |
| 过程检查 | 用 Motion Viewer 比较本批次各阶段的动作 |

Viewer 用于检查批次和中间结果；网页中能播放不等于质量门禁通过，也不代替 Godot 中的最终蒙皮和运行验证。

## 快速开始

从选定的父目录克隆仓库：

```bash
git clone https://github.com/cc8887/Kimodo-Pipeline.git
cd Kimodo-Pipeline
```

生产需要 Kimodo 环境、已获授权的 Llama 文本编码器访问权限、原版 ALS 参考数据，以及配置好的 Godot 消费项目。仓库克隆本身不包含这些模型和资产。请先按[从零入门教程](docs/入门教程.md)配置 WSL2 和 Kimodo；在激活的 Kimodo Python 环境中安装本工具，并按教程创建 Git 忽略的本机配置文件：

```bash
python -m pip install -e .
```

配置完成后，在 WSL Bash 的仓库根目录查看跑步生产范围，并运行所选范围：

```bash
python -m tools.doctor --verify-reference
python tools/produce_als_resources.py inspect --scope run
python tools/produce_als_resources.py run --scope run --keep-going
```

`run` 是跑步动作子集，不代表完整非道具发布范围；选定范围是否可交付，以终端退出码和批次报告中的门禁结果为准。完整流程和其他 `scope` 见教程。

### Motion Viewer

在 Windows PowerShell 的独立终端中运行：

```powershell
& .venv-win\Scripts\python.exe tools\start_motion_viewer.py --port 10016
```

然后打开 [http://127.0.0.1:10016/](http://127.0.0.1:10016/)。此 Windows Python 环境按入门教程建立。批次检查步骤见教程中的“每次生产后，启动 Motion Viewer 检查”。

## 修改 Prompt

多数动作提示词按生产范围配置：

| 范围 | 主要文件 |
| --- | --- |
| Run | `tools/kimodo_pipeline/run_rules.json` |
| Crouch | `tools/kimodo_pipeline/crouch_rules.json` |
| Walk、方向步态、默认非道具范围 | `tools/kimodo_pipeline/directional_gait_rules.json`、`tools/kimodo_pipeline/non_prop_rules.json` |
| Air、Mantle 和部分固定分支 | `tools/kimodo_pipeline/air.py`、`graph.py` |

具体规则、继承关系和安全试改流程见[入门教程的 Prompt 章节](docs/入门教程.md#6-想改-prompt到哪里改)。计划生成的 `graph.json` 中 `params.prompt` 字段可用于确认实际生效的文案。

## 文档

| 文档 | 内容 |
| --- | --- |
| [从零入门教程](docs/入门教程.md) | Kimodo、Llama 权限、WSL/CPU 方案、生产、Prompt、Viewer 和 Git 操作 |
| [文档索引](docs/README.md) | 架构、算法、验证记录和许可文档入口 |
| [独立目录契约](docs/architecture/standalone-workspace.md) | 参考快照、制作包、Godot 部署副本和验收边界 |
| [生产算法说明](tools/kimodo_pipeline/README.md) | 生产依赖图、约束、派生和质量门禁 |
| [来源与许可边界](docs/architecture/source-and-license-boundaries.md) | 本仓库代码、原工程代码、资产与第三方依赖的许可范围 |
| [迁移验证记录](docs/verification/2026-10-11-independent-workspace.md) | 本次独立目录迁移的验证范围及未覆盖项 |

## 仓库与资源

- `tools/kimodo_pipeline/`：生产规则、算法与回归测试。
- `tools/motion-viewer/`：Motion Viewer 服务和前端。
- `config/`：可提交的配置模板。
- `docs/`：入门教程、架构契约与验证记录。
- `workspace/`：本地参考快照、生成包、运行回执和日志，均由 Git 忽略。

模型权重、访问令牌、本机路径、原版 ALS 参考数据及生成资源不会随仓库分发。将代码检出到另一台电脑不等于已取得上述资产的使用权；资产权利以来源方许可为准。

## 许可

本 Pipeline 自有代码和相关文档采用 [PolyForm Noncommercial License 1.0.0](LICENSE.md)。该许可不追溯撤销原有 MIT 授权，也不替代 Kimodo、模型、原版 ALS 资源或其他第三方内容各自的许可。分发时请保留根目录许可证和声明文件；详情见 [NOTICE](NOTICE)、[ASSET_LICENSE.md](ASSET_LICENSE.md) 与 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
