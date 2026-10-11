# Kimodo ALS Pipeline

独立的 Kimodo 动作生产工具：依赖图生成、母动作及派生、纯脚本 ALS 重定向、有界 IK、Root/Pelvis 分解、逐资产曲线、最终 FBX 回读和 Godot 运行验证。

**第一次使用请从 [入门教程](docs/入门教程.md) 开始。** 教程包含 Windows/WSL 终端区分、官方 Kimodo 下载、Llama 权限申请、模型预下载、CPU 方案地址、日常生产、prompt 修改和每批 Motion Viewer 检查。

**本 Pipeline 自有代码和文档采用 PolyForm Noncommercial 1.0.0，仅授权非商业用途；商业使用须取得权利人另行授权。** 许可正文见 [LICENSE.md](LICENSE.md)，具体范围见 [NOTICE](NOTICE)。

## 本机快速使用

在本目录打开 PowerShell：

```powershell
& .venv-win\Scripts\python.exe -m tools.doctor --verify-reference
& .venv-win\Scripts\python.exe tools\produce_als_resources.py inspect --scope run
& .venv-win\Scripts\python.exe tools\start_motion_viewer.py --port 10016
```

打开 [Motion Viewer](http://127.0.0.1:10016/)。Viewer 的批次登记读取独立生产目录，当前项目对照读取配置的 Godot 项目。

真正生成需要 Kimodo Python 环境：

```powershell
$KimodoPython = (& .venv-win\Scripts\python.exe -c "from tools.kimodo_pipeline.paths import configured_path; print(configured_path('python'))")
& $KimodoPython tools\produce_als_resources.py run --scope run --keep-going
```

默认完整执行；质量失败返回非零退出码并保留证据。独立版本在 `workspace/als/assets/generated/`；Godot 适配器复制到消费项目的同名新目录，随后执行原有构建、导入、六域绑定及场景检查。默认指针启用仍走原有门禁。

## 目录

| 路径 | 内容 |
|---|---|
| `tools/kimodo_pipeline/` | 生产算法、规则和回归测试 |
| `tools/motion-viewer/` | 只读网页与浏览器测试 |
| `config/` | 可提交的配置模板；实际路径写根目录的 local TOML |
| `docs/architecture/` | 稳定制作与交付契约 |
| `workspace/als/` | 忽略入 Git 的原始参考快照、生产包和 Viewer 登记 |
| `workspace/runs/` | 每次独立的 DAG、回执、失败和运行日志 |

架构见 [独立目录契约](docs/architecture/standalone-workspace.md)。旧工程的未提交生产代码已经按当前字节保存来源记录；此目录是迁出后的工具工程，旧项目代码和资产保持以便继续原有工作。

文档索引见 [docs/README.md](docs/README.md)，本次测试、真实导入和未覆盖项见 [迁移验证记录](docs/verification/2026-10-11-independent-workspace.md)。

资产范围与第三方条款见 [ASSET_LICENSE.md](ASSET_LICENSE.md) 和 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。原工程三份声明与署名完整保留在 [licenses/source/](licenses/source/README.md)，其既有 MIT 授权与第三方许可继续适用于原材料。独立目录中的适用范围见 [来源与许可边界](docs/architecture/source-and-license-boundaries.md)。

此 Git 仓库当前只有本地历史，没有配置远程。安装使用源码 editable 模式；wheel/离线安装包尚未作为交付形式验证。
