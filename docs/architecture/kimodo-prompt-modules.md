# 可替换风格的动作 Prompt 模块

`tools/kimodo_pipeline/prompt_modules.py` 只编译提示词和对照实验计划，不生成动作、不改变数值门禁，也不判断生成动作是否合格。规则通过 `promptModules` 显式启用；未列出的动作保留已有配方、`motionPromptOverrides`、合同草稿之间的优先级和原始文案字节。

## 功能意图与风格分开

| 模块 slot | 职责 | 示例 |
|---|---|---|
| `action` | 动作种类，必须且只能有一个 | `is walking`、`is sprinting` |
| `direction` | 移动方向与面向 | backward、向左移动而面向前方 |
| `phase` | 稳态周期或过渡的语义 | complete alternating strides |
| `stance` | 站立高度、蹲伏等基本状态 | in a low crouch |
| `posture` | 可替换的躯干、头颈风格 | 直背平衡头部、轻微前倾 |
| `arms` | 上臂、小臂、手的可独立要求 | 上臂稳定、小臂摆动 |
| `style` | 简短、可观察的运动质感 | smooth graceful movement |

模块按上述顺序确定性拼接，主体默认为 `A person`，动作动词紧接主体。模块内部写英文短语，避免故事、摄影词和重复关节枚举。词句优化原则来自 [Kimodo 官方指南](https://research.nvidia.com/labs/sil/projects/kimodo/docs/key_concepts/limitations.html)：中等细节、每段自足、聚焦一两个行为。精确接触、轨迹、时间和关节目标继续由数值约束承担。

动作模块的 `claims` 声明 `action`、`direction`、`heading`、`phase`、`stance`、`contact`；风格模块不能改写这些功能维度。风格可声明 `torsoPose`、`upperArmMotion`、`forearmMotion` 等可比较要求。同一维度的冲突值在生成前拒绝。例如上臂同时 `steady` 与 `swinging` 无效；上臂 `steady`、小臂 `swinging` 有效。

这是结构化意图检查，不是英语自然语言推理。手工把文案改成“前进”、却继续声明 `direction: backward`，编译器不能识别这种失配；最终生效文案必须在计划中展示，真实方向仍由运动学门禁验证。原有门禁不能因为 `claims` 声明正确而省略。

`when` 仅按功能维度选择风格模块。示例 `upright-standing` 只在 `stance: standing` 时生效，`inclined-crouch` 只在 `stance: crouching` 时生效，因此蹲伏不会继承直立上身。`requires` 表达显式模块依赖。被条件略过的模块列入回执 `omittedModules`，不借用其他姿态的成功证明。

## 规则与兼容入口

[prompt_module_examples.json](../../tools/kimodo_pipeline/prompt_module_examples.json) 是继承现有方向步态规则的示例。它包括 `neutral`、`feminine`、`athletic` 三种可替换风格，明确列出 Walk、Run、Sprint 和部分蹲走；不默认改变 Jump、Mantle、Air。

当前 `Sprint_Acceleration` 的原生参考是持续运动周期。因此示例仍声明 `phase: steady`，文案使用 `at a sustained fast pace`，没有因为资产名字含 `Acceleration` 就把它改成从站立起跑。真正的起跑动作应另建过渡合同和 phase 模块，不能沿用周期门禁。

修改一类内容有三种粒度：

1. 改某个 `modules.<id>.text`：所有引用该模块的动作同步变更。
2. 改 `style` 或某个 `motions.<semantic>.style`：替换全局或单动作风格，功能模块保持。
3. 配方 `base` 加 `remove` / `add`：例如后退 Walk 从前进 Walk 移除 `forward`、添加 `backward`。移除不存在的模块、重复模块和继承环都会拒绝。

生产节点接入只需要替换 prompt 选择位置，并把回执放进参与节点身份计算的 `params`：

```python
from tools.kimodo_pipeline.prompt_modules import resolve_prompt

compiled = resolve_prompt(
    rules, row['semanticName'], row['promptDraft'], explicit_prompt=recipe_prompt,
    expected_claims=constraint_intent,
)
params['prompt'] = compiled['prompt']
if compiled['receipt'] is not None:
    params['promptModules'] = compiled['receipt']
```

`constraint_intent` 可为 `None`，或由实际数值约束提供如 `{'direction': 'backward', 'phase': 'steady'}`。有声明却不匹配、或 prompt 缺少该声明时拒绝；不要仅从动作字符串猜测。配置显式覆盖的动作优先使用模块化结果，包括旧配方中硬编码的 prompt；未启用动作的旧优先级完全保留。

`compile_prompt(config, semantic, style=..., add_modules=..., remove_modules=..., expected_claims=...)` 可独立用于计划预览。回执包含有效模块、条件略过项、合并后的意图、最终 prompt 和 SHA-256。有效模块按固定 slot、ID 排序，字典顺序、列表顺序或未使用模块的变化不改变有效 prompt 指纹。真正改变文本、有效意图或外部约束意图会改变指纹。模型、seed、CFG、真实约束载荷等仍应由完整生成节点身份一起绑定。

## 风格不能被后续阶段覆盖

文本风格、生成约束和后处理必须指向同一风格意图。特别是旧 `locomotionPosture` 会把结果强制改造成指定的直背／稳大臂风格；只替换成 athletic 文案而保留该后处理，不能算 athletic 生产或有效对照。

每次换风格都需要检查：

- `locomotionPosture` 是否关闭，或选择了匹配本次风格的独立 profile。
- `gaitClosure.*.upperBodyIdleReference`、手朝向约束与完整身体关键帧是否意外固定旧上身风格。
- 下肢支撑/根轨迹约束是否可独立复用；若只需要下肢功能骨架，应使用明确的受约束关节范围，不把原生上身所有关节都钉死。
- 风格专属验收是否仅用于对应 profile：大臂稳定是当前 feminine 的要求，不能作为 athletic 或所有后续风格的普适门禁。

通用门禁继续验证语义、可用支撑、脚滑、骨长、周期接缝、Root/Pelvis、FBX 回读和消费者运行。风格验收独立报告，可用 Idle 或风格参考检查躯干、上臂、小臂和手的变化。不能为了某一风格降低功能门限，也不能用功能通过替代风格保真。

## 最小对照与可重复实验

先对失败动作做同 seed 的配对实验，单次只改一个模块或整套明确标记的风格选择；再用其他 seed 验证稳定性。整个风格切换是一个配置因素，但内部可能包含多个词句，不能据此声称某一形容词独立提高成功率。想回答具体原因，应逐模块删除，例如分别删除 `feminine-quality`、`upright-standing` 或 `steady-upper-arms`，保留 `moving-forearms`。

```python
import json
from pathlib import Path
from tools.kimodo_pipeline.prompt_modules import build_ablation_plan

config = json.loads(Path('tools/kimodo_pipeline/prompt_module_examples.json').read_text())['promptModules']
plan = build_ablation_plan(
    config, 'Sprint',
    generation={
        'model': 'Kimodo-SOMA-RP-v1.1', 'skeleton': 'SOMA77',
        'seconds': 4, 'fps': 30, 'steps': 100, 'cfgWeights': [2.0, 2.0],
        'constraintFingerprint': 'REPLACE_WITH_ACTUAL_CONSTRAINT_PAYLOAD_SHA256',
    },
    seeds=[11, 23], styles=['neutral', 'athletic'],
    remove_modules=['steady-upper-arms'],
)
```

生成计划给每种变体同一组 seed，冻结完整 `generation` 字典，标记 `status: planned`，并记录控制项及各行指纹。示例约束指纹必须换成实际约束载荷 SHA-256；无约束时明确 `None`。真正运行时必须保留这些控制项，记录模型版本/权重身份和代码版本，输出到新的证据目录。不要把改 prompt 又改 seed、CFG、约束、时长的结果归因于文本。

每个实验至少记录：原文与模块、seed/设置/约束哈希、源语义与支撑结果、修复量和被改变关节、最终目标骨架检查、最终 FBX 回读以及 Godot 实测。失败包括未生成、门禁不支持、缺失、源不合格和后处理不合格，均进入分母。只有不同风格经过这些实际阶段，才能报告跨风格的成功率。

当前模块自身的受控测试覆盖兼容优先级、功能/风格冲突、蹲伏条件、单模块删除、指纹稳定、数值意图冲突、配对实验控制和输入错误；这些测试不代表三个风格的模型生成或最终资源已经通过。
