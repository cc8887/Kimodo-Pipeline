# 来源与许可边界

本工具从 Godot ALS 工程提取。独立 Pipeline 自有集成代码、脚本、测试、规则和相关文档采用 **PolyForm Noncommercial 1.0.0**；根目录 [LICENSE.md](../../LICENSE.md) 使用[官方完整文本](https://polyformproject.org/licenses/noncommercial/1.0.0)，具体范围和署名见 [NOTICE](../../NOTICE)。仅授权许可条款规定的非商业用途，商业使用须权利人另行授权。

原工程的三份声明和署名原样保留在 [licenses/source/](../../licenses/source/README.md)，与源工程文件逐份 SHA256 一致。原工程 MIT 授权继续适用于按其条款取得的原材料；新项目许可不能追溯取消这项授权，也不能改变第三方代码或上游发布版本的许可。逐文件代码来源仍见 [extraction-source.json](../verification/extraction-source.json)。

资产、导出数据以及转换或生成后的资产按根目录 [ASSET_LICENSE.md](../../ASSET_LICENSE.md) 和相应来源条款处理；第三方范围见 [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md)。把文件放入独立仓库、改成 JSON，或用本工具处理，都不改变其权利来源。非商业代码许可也不自动授予第三方资产再分发权限。

原声明列举的资产目录在此仓库对应如下。表中是目录边界说明，不额外授予使用或再分发权限。

| 独立目录 | 内容与声明范围 |
|---|---|
| `workspace/als/assets/config/`、`workspace/als/assets/generated/als_v4/` | 从消费项目保存的原始配置、骨架、曲线和动画参考；由 Git 忽略，仍属于原声明的资产范围 |
| `workspace/als/assets/generated/als_nonprop_<版本>/` | 转换、重定向与生成包；许可范围取决于原始数据及模型等来源 |
| `workspace/runs/` 中的动作、预览和资产夹具 | 忽略入 Git 的生产与诊断制品；Git 忽略规则不提供资产授权 |
| `docs/architecture/kimodo-als-asset-contracts.json`、`docs/architecture/kimodo-als-asset-catalog.md` | 来源于原项目的资产描述数据；不能仅凭文本格式将导出内容视为项目源代码 |

Kimodo、Llama、LLM2Vec、模型权重、ufbx 和 Motion Viewer 的 three.js/loader 等外部材料，各自保留上游许可与适用条款。源码、模型访问权限和资产使用权限分别来自各自来源；本仓库的三份声明不替代这些条件。Viewer 中已有第三方文件的来源与许可头保持原样。

2026-10-11 的敏感信息修复记录对应当时三份根目录声明原样迁入的事实；随后按用户要求设置当前非商业许可，并把来源声明完整移到 `licenses/source/`。历史扫描与许可校验不自动适用于后续文件；本次验证见[非商业许可记录](../verification/2026-10-11-noncommercial-license.md)。操作入口见[入门教程](../入门教程.md)，制作与消费目录的技术边界见[独立目录契约](standalone-workspace.md)。
