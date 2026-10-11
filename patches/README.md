# 可选的本地 Llama base 路径覆盖

`kimodo-local-llama.patch` 保存当前 Windows Kimodo 的两个文本编码来源改动，基于上游 commit `58e781898b3d7e328a676a75d3e338c45dce3ad9`。它允许 `KIMODO_LLAMA_BASE_MODEL_PATH` 指定本地 base 权重，再应用原有 MNTP 和 supervised 适配器。

标准 HF 缓存安装不要求该补丁。本机当前后端已经包含这些改动，不要再次应用。需要复现本地权重目录布局时，在匹配版本的 Kimodo 仓库中先检查再应用：

```bash
git apply --check "$KIMODO_PIPELINE_ROOT/patches/kimodo-local-llama.patch"
git apply "$KIMODO_PIPELINE_ROOT/patches/kimodo-local-llama.patch"
```

`KIMODO_PIPELINE_ROOT` 在[入门教程](../docs/入门教程.md#3-下载和安装-kimodowsl-方案)中从工具仓库当前目录取得；本段命令在 Kimodo 后端目录执行。

`--check` 失败时保留现有修改并检查上游差异；不能强制覆盖编码器文件。该改动只改变权重来源加载路径，不将 Llama 授权流程替换为其他模型。
