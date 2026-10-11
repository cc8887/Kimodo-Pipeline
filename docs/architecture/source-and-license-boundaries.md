# 来源与许可边界

本工具从 Godot ALS 工程提取。根目录 [LICENSE.md](../../LICENSE.md)、[ASSET_LICENSE.md](../../ASSET_LICENSE.md) 和 [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md) 保留源工程的完整原文和署名；本次迁入时三份文件的磁盘 SHA256 与源文件一致。逐文件代码来源仍见 [extraction-source.json](../verification/extraction-source.json)。

`LICENSE.md` 的 MIT 范围是项目自有代码与相关文档，第三方代码仍采用自己的许可。资产、导出数据以及转换或生成后的资产按 `ASSET_LICENSE.md` 和相应来源条款处理；把文件放入独立仓库、改成 JSON，或用本工具处理，都不改变其权利来源。

原声明列举的资产目录在此仓库对应如下。表中是目录边界说明，不额外授予使用或再分发权限。

| 独立目录 | 内容与声明范围 |
|---|---|
| `workspace/als/assets/config/`、`workspace/als/assets/generated/als_v4/` | 从消费项目保存的原始配置、骨架、曲线和动画参考；由 Git 忽略，仍属于原声明的资产范围 |
| `workspace/als/assets/generated/als_nonprop_<版本>/` | 转换、重定向与生成包；许可范围取决于原始数据及模型等来源 |
| `workspace/runs/` 中的动作、预览和资产夹具 | 忽略入 Git 的生产与诊断制品；Git 忽略规则不提供资产授权 |
| `docs/architecture/kimodo-als-asset-contracts.json`、`docs/architecture/kimodo-als-asset-catalog.md` | 来源于原项目的资产描述数据；不能仅凭文本格式将导出内容视为 MIT 代码 |

Kimodo、Llama、LLM2Vec、模型权重、ufbx 和 Motion Viewer 的 three.js/loader 等外部材料，各自保留上游许可与适用条款。源码、模型访问权限和资产使用权限分别来自各自来源；本仓库的三份声明不替代这些条件。Viewer 中已有第三方文件的来源与许可头保持原样。

本次只恢复随提取源码应保留的声明并说明目录映射，不重新许可 ALS、Epic/Fab、Mixamo 等第三方资产，也不将模型下载成功视为资产再分发许可。操作入口见[入门教程](../入门教程.md)，制作与消费目录的技术边界见[独立目录契约](standalone-workspace.md)。
