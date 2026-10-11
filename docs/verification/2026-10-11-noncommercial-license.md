# 2026-10-11 Pipeline 非商业许可

按用户要求，独立 Pipeline 自有集成代码、脚本、测试、规则及相关文档采用 **PolyForm Noncommercial 1.0.0**，商业用途须权利人另行授权，具体许可用途以标准条款为准。根目录 `LICENSE.md` 使用[官方完整文本](https://polyformproject.org/licenses/noncommercial/1.0.0.txt)，`NOTICE` 保留原有版权署名并说明项目范围。README、入门教程、资产说明、第三方说明和架构边界已同步。

原工程三份声明原样保留到 `licenses/source/`，本次磁盘文件 SHA256 与源工程逐份一致。新项目许可不追溯取消原材料的 MIT 授权，也不替换 NVIDIA Kimodo、模型和第三方组件的条款。前一份敏感信息修复记录对应当时的文件与提交，不改写其历史结果。

`pyproject.toml` 声明 SPDX 标识 `PolyForm-Noncommercial-1.0.0`，通过 `project.license-files` 包含根目录四份许可/声明和来源目录四份文件；build backend 最低版本为 `setuptools>=77`，以支持这两个标准元数据字段。依据见 [setuptools 官方配置文档](https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html)。

本次证据另存于被 Git 忽略的 `workspace/license-noncommercial-20261011/`：

| 核查 | 证据 |
|---|---|
| 许可正文与官方下载字节一致 | `preservation-baseline.json`、`validation.json` |
| 三份原声明及原署名保留 | `licenses/source/` 与源工程 SHA256 比较，见 `validation.json` |
| wheel 许可元数据及声明文件 | `wheel-build.log`；检查 `License-Expression` 和全部 8 个 `License-File` 及其正文 |
| 当前源码 editable 安装元数据 | `editable-refresh.log`、`validation.json`；只重装本工具，不更新运行依赖 |
| 当前入口 | `inspect-run.json`；`inspect --scope run` 选择 33 个节点 |
| 敏感内容、文档链接、改动边界 | `validation.json`、`diff-check.log`；扫描当前 tracked 文件及本次新增文件 |
| 原有 JSON 与本地配置 | 10 份 tracked JSON 与索引字节一致，实际 local 配置 SHA256 保持 |

wheel 检查只覆盖许可元数据与声明携带，不作为完整 wheel/离线运行交付验收。生产代码、规则 JSON 和资产未修改；本次不运行模型生成或 Godot 整链场景。许可调整按本会话的独立 Git 管理要求保存为独立本地提交，未推送；最终提交、范围和工作区状态见 `git-state.json`。前一轮敏感信息修复的提交及证据保留为历史记录。
