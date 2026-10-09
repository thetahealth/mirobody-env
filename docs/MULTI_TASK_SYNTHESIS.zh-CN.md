# 跨任务合成与验证

[English](MULTI_TASK_SYNTHESIS.md) · [README](../README.zh-CN.md)

新合成任务先定义金标与作答契约，再组合已有任务中的病人时间线、噪声、干扰事件和事件密度控制。验证需要覆盖组合后的记录：复用前提仍成立的公共校验，并新增检查，确保任务金标能对应到 agent 实际可见的证据。

## 案例：罕见病表型编码

[罕见病扩展](https://github.com/thetahealth/mirobody-env/tree/rare-code-bench/haenv_rare)位于 `rare-code-bench` 分支，未包含在主分支发布的 `haenv` 包中。任务要求 agent 编码人类表型本体（HPO）术语，区分阳性与阴性、患者与亲属，并给出 Orphanet 诊断；适用时还需识别致病基因与变异。

| 组成部分 | 案例中复用或新增的能力 |
|---|---|
| 病人记录 | 复用病人画像、截断时点与病程长度生成，在现有代谢病人世界上叠加罕见病表现。 |
| 事件密度 | 复用测量、症状与生活事件的默认密度设置；本案例未演示密度扫描实验。 |
| 解读难度 | 增加阴性表现、归属于亲属的表现和叙述变体，与框架已有的噪声、干扰控制配合。 |
| 验证 | 复用公共事件过滤与性别一致性规则；新增金标术语在可见记录中的可恢复性、VCF 变异读回、附件文本答案泄漏等检查。 |
| 评测 | 通过插件接口增加编码判据，与已有任务判据共同使用。 |

复用的画像、时间线和密度函数见分支中的[任务生成器](https://github.com/thetahealth/mirobody-env/blob/rare-code-bench/haenv_rare/haenv_rare/gen_job.py)；采样与验证见[表型生成器及闸门](https://github.com/thetahealth/mirobody-env/blob/rare-code-bench/haenv_rare/haenv_rare/gen.py)。

## 为什么要验证组合后的结果

早期开发中，公共事件过滤规则删掉了含禁用词的症状句，但对应的 HPO 术语仍留在金标中。罕见病新增的可恢复性检查发现了证据缺失。公共性别一致性检查还发现了男性病例中的月经相关表现；附件读回检查发现了未实际写入 VCF 的预定变异。这些问题需要在生成阶段修复，再交给模型评测。

公共校验的前提也可能不适用于新任务。批次级 `footprint_discriminates_real_symptom` 检查假设症状来自症状主题注册表，而 HPO 采样的表现不在其中。罕见病流程显式覆盖这道批次门，并在 `batch.json` 记录；逐例校验仍然执行。任务生成器写明了这一例外，不能据此宣称所有公共审计均已通过。

## 如何使用现有技能

- [haenv-synth](../skills/haenv-synth/SKILL.md)：配置生成、构建并验证病例、查看拒绝原因。
- [haenv-extend](../skills/haenv-extend/SKILL.md)：通过插件增加世界组件和判据。
- [外部任务契约](design/external-task-contract.md)：连接任务金标、agent 可见证据和检查。

这些技能共同支持该流程。增加任务仍需选择兼容的组件并验证组合结果，各项现有校验的适用性需要逐项确认。
