<h1 align="center">Health Agent Environment</h1>

<p align="center">
  <strong>生成与评测，一条流程贯通</strong><br>
  生成合成病人与临床场景，再依据代码推导的标准答案评测健康智能体。
</p>

<p align="center">
  <a href="https://github.com/thetahealth/mirobody-env/blob/main/LICENSE"><img src="https://img.shields.io/badge/code-MIT-blue.svg" alt="Code: MIT"></a>
  <a href="https://github.com/thetahealth/mirobody-env/blob/main/LICENSE-DATA"><img src="https://img.shields.io/badge/data-CC%20BY%204.0-green.svg" alt="Data: CC BY 4.0"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-3776AB.svg?logo=python&logoColor=white" alt="Python 3.10+"></a>
  <a href="https://pypi.org/project/haenv/"><img src="https://img.shields.io/pypi/v/haenv.svg" alt="PyPI"></a>
  <a href="https://github.com/thetahealth/mirobody-env/actions/workflows/ci.yml"><img src="https://github.com/thetahealth/mirobody-env/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://thetahealth.github.io/mirobody-env/"><img src="https://img.shields.io/badge/demo-live-black.svg" alt="Live demo"></a>
</p>

<p align="center">
  <a href="https://github.com/thetahealth/mirobody-env/blob/main/README.md">English</a> &middot; <strong>简体中文</strong>
  &nbsp;|&nbsp;
  <a href="https://thetahealth.github.io/mirobody-env/">在线演示</a> &middot;
  <a href="#四个题包">题包</a> &middot;
  <a href="#合成病人">合成病人</a> &middot;
  <a href="#快速开始">快速开始</a> &middot;
  <a href="#工作原理">工作原理</a> &middot;
  <a href="#计分">计分</a> &middot;
  <a href="#排行榜">评测结果</a> &middot;
  <a href="#评测你自己的-agent">评测你的 agent</a> &middot;
  <a href="#agent-skills">技能</a> &middot;
  <a href="#引用">引用</a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/thetahealth/mirobody-env/main/docs/figures/readme_hero.png" alt="HAEnv 五步：定义隐藏的病人状态，渲染多年的记录，发布检查，被测 agent 回答题目，代码与 LLM 裁判对照由同一状态推出的隐藏金标判分" width="100%">
</p>
<p align="center"><sub>agent 只接收截至 <code>T</code> 的记录，隐藏金标只交给判分器；判分器是代码，另有一个 LLM 裁判判定处理方案质量 <code>Q</code>（见<a href="#计分">计分</a>）。</sub></p>

## 测什么

HAEnv（Health Agent Environment）的代码仓库是 `mirobody-env`，安装后的 Python 包与命令行工具名为 `haenv`。

HAEnv 评测在随时间展开的病历上做出的临床决定。合成病人的病历按月增长，agent 看到截止时点 `T` 之前的记录，做出一个决定。正确答案由渲染这份病历的世界在写出病历之前就已确定，对 agent 隐藏。

HAEnv 负责生成病人、设定难度、推出金标。要用已发表的健康基准（如 ESL-Bench）评测现成系统，请用
[mirobody-eval](https://github.com/thetahealth/mirobody-eval)。

### 四个题包

基准由四个任务题包组成，每道题考一个核心决定。

| 题包 | `T` 处的决定 | 答案 | 判分方式 |
|---|---|---|---|
| ① 限额诊断 | 哪些病能解释这份病历——第二种病藏在被第一种病解释掉的表现背后；是否需要临床复核 | 鉴别诊断、要开的检查、复核标记 | 代码对世界真值判：按登记的别名把列出的诊断与金标比对，不调用模型 |
| ② 分诊 | 病人现在该去哪 | `ed_now` · `within_24h` · `routine_followup` · `watchful_waiting` | 代码对世界真值判 |
| ③ 用药调整 | 这次随访时药该怎么办 | `uptitrate` · `downtitrate` · `maintain` · `switch` · `check_adherence_or_adverse_effect` | 代码对世界记录的剂量线、取药与控制目标判 |
| ④ 化验变化归因 | 与上次结果相比的变化是不是真的 | `true_change` · `analytic_biological_noise` · `preanalytical` · `method_difference`，以及真变化的幅度 | 代码对世界的真值与逐读数扰动记录判 |

1.x 榜的两条诊断轨 `ddx-timeline` 与 `ddx-workup` 作为桥接轨保留在仓内：题包 ① 带有取自它们的桥接题，它们是 1.x 榜与各题包之间唯一的联系。

- **先定金标。** 疾病、药物反应、依从性、化验真值与测量噪声在渲染病程之前确定，各题包的金标由代码从它们推出。
- **病人内部自洽。** 体重、化验、用药、可穿戴数据流和生活事件由同一套规则在同一条时间线上生成。已知共病会进入世界的表，所以在题包 ③ 和 ④ 里，它会在该改变答案的地方改变正确答案。
- **病历读起来像病历。** 主诉是病人自己的口语说法，病历列出病人已知的病以及每种病的用药。
- **难度是参数。** 测量伪影、干扰事件、临床转折的时点都是 job 文件里的设置。
- **安全失误不会被平均掉。** 硬性门槛（包括越权改药、编造证据、漏掉急症、过早下结论）命中任何一道，该题记零分，其他得分无法抵消。过度分诊与漏请临床复核这两道倾向门照常判出并报告，但不进乘子。

| 术语 | 含义 |
|---|---|
| 截断时点 `T` | 题面只含 `T` 之前的记录；判分用 `T` 之后的部分或世界已定的真值 |
| 隐变量 | 世界在渲染之前定下的隐藏状态：疾病、药物反应、依从性、真值、噪声 |
| 题包 | 一个任务的题集，由生成器按 job 文件出题、经题包审计检查 |
| 种子 | 决定题包抽哪些病例、各答案类别如何分配；题包只记种子的 sha256 |
| 出题闸门 | 生成的病例发布前必须通过的检查：前提校验、逐项校验、泄漏探针 |
| 硬性门槛 | 使所在计分单元记零分的安全失误 |
| 批次 | 一次运行的目录：病例、回答、得分与指纹 |

### 题包是管线

题包不是一份固定文件。题量和种子由你定，一条命令写出 job、用确定性生成器出包，并跑所有题包共用的出题期审计，全程不调用模型：

```bash
uv run --with scikit-learn --with joblib python tools/generation/make_pack.py --pack p4 --n 50 --out packs/p4
```

`--pack` 取 `m2`（①）、`p2`（②）、`p3`（③）或 `p4`（④）（`pack2` 是 1.3.0 榜的分诊包）；`--n` 是题量（50、100……）；`--seed`（或 `HAENV_PACK_SEED`）设种子。同一题包、同一题量、同一种子出的包相同。job 带 `pack: {n_items, seed_sha256}` 头，`batch.json` 记录题量、种子的哈希和 job，从不记种子本身。任何题量下各类占比都按该题包的比例表分配。

| 题包 | 示例 job（当前默认设置） | N = 50 时的题数 | N = 50 时的答案类别 |
|---|---|---:|---|
| ① 限额诊断 | `benchmarks/inputs/m2-pack1.job.yaml` | 50 | 34 道新题（24 道有隐藏病、10 道没有）与 16 道桥接题 |
| ② 分诊 | `benchmarks/inputs/pack2-triage.job.yaml` | 50 | `ed_now` 13 · `within_24h` 13 · `routine_followup` 12 · `watchful_waiting` 12 |
| ③ 用药调整 | `benchmarks/inputs/p3-meds.job.yaml` | 50 | `maintain` 12 · `downtitrate` 12 · `check_adherence_or_adverse_effect` 12 · `uptitrate` 8 · `switch` 6 |
| ④ 化验变化归因 | `benchmarks/inputs/p4-followup.job.yaml` | 50 | `true_change` 13 · `analytic_biological_noise` 13 · `preanalytical` 12 · `method_difference` 12 |

- **示例 job，不是评测题集。** 表中 job 文件是当前默认设置下的示例（公开种子、N = 50）。照着出得到同样设计的包，而不是 1.4.0 榜的题；后者以冻结题包文件发布（见[排行榜](#排行榜)）。② 行是 1.3.0 的分诊 job `pack2-triage`；1.4.0 的分诊任务用 `--pack p2` 生成。
- **种子。** 榜的批次只记种子的哈希。1.4.0 榜用公开种子（见[排行榜](#排行榜)）。更早的榜用私有种子，从仓库无法重出其题；换榜时公开旧榜的种子，旧榜随之可复现。
- **审计是发布闸门。** `tools/validation/pack_audit.py` 检查出好的批次：job 计划的每个格子都在，从记录重新推出的金标与存下的金标一致，不看题面的桩与表面特征预测不了答案。审计没过，`haenv run` 拒绝评测该包的题。

桥接轨和两个较小的示例任务以普通 job 文件发布：

| 任务 | job 文件 | 病例规格数 | 形式 |
|---|---|---|---|
| 多时点鉴别诊断（桥接） | `benchmarks/inputs/ddx-timeline.job.yaml` | 145 | 在多个时点分别提问 |
| 限预算开检查（桥接） | `benchmarks/inputs/ddx-workup.job.yaml` | 145 | agent 在预算内自行开检查 |
| 体重回升预测与驱动因素归因 | `benchmarks/inputs/early_warning-20.job.yaml` | 20 | 在 `T` 处提问一次 |
| 多轮随访复盘 | `benchmarks/inputs/tracking_review-20.job.yaml` | 20 | 分轮推进，agent 可修正 |

规格通过出题闸门才会成为病例。题包、病例数和已知缺口见[数据卡](https://github.com/thetahealth/mirobody-env/blob/main/docs/DATA_CARD.md)。

**一条流水线，多个任务。** 一个任务 = 一种金标形状 + 一份作答契约。各题包的病人由同一个世界渲染、过同一道逐项校验的出题闸门，每个题包把自己的金标与计分登记为一个判据组。自己加一个任务类型不需要改本仓，而是一个仓外的包：见[评测你自己的 agent](#评测你自己的-agent) 与 [`docs/design/external-task-contract.md`](https://github.com/thetahealth/mirobody-env/blob/main/docs/design/external-task-contract.md)。

**语言。** 题面与病例内容为中文：指令部分，以及病历中的自由文本字段（主诉、情境、事件）。字段名、数据流名、答案枚举值与各类编号为英文。

## 合成病人

[在线演示](https://thetahealth.github.io/mirobody-env/)在浏览器里生成病人，拖动参数即可改变病程、化验与依从性。

<p align="center">
  <img src="https://raw.githubusercontent.com/thetahealth/mirobody-env/main/docs/figures/readme_cohort.png" alt="三幅图：替尔泊肽维持者与司美格鲁肽低应答者各 60 人的个体药效分布；90 天内 HbA1c 变化与空腹血糖变化的散点；12 个病人按各自的 T 对齐的体重轨迹，在 T 之后分化为回升与维持" width="100%">
</p>
<p align="center"><sub>（a）两份规格，各换 60 个病例编号。每个病人在其声明的驱动因素所允许的区间内（阴影）抽取个体药效。
（b）前 90 天内，维持者的空腹血糖与 HbA1c 一起下降；低应答者两项都几乎不动。（c）<code>early_warning-20</code> 的 20 条规格中通过出题闸门的 12 例，各按自己的 <code>T</code> 对齐：
回升与维持大多在 <code>T</code> 之后分开，在 <code>T</code> 时做预测必须依靠更早的信号。</sub></p>

<p align="center">
  <img src="https://raw.githubusercontent.com/thetahealth/mirobody-env/main/docs/figures/readme_difficulty.png" alt="同一个病人的三档设置：默认；加两处体重瞬时尖峰并记为陷阱；再加高干扰档，T 之前的干扰事件从两个变成七个" width="92%">
</p>
<p align="center"><sub>同一个病人的三档难度。底层序列完全相同（灰点标出默认档中被改动的位置），结局与驱动因素不变。
每处瞬时尖峰在金标里记为一个陷阱：多轮形式中，agent 若在伪影处把风险调高，会被扣逆转跟踪分；单问形式中，若在未标记可疑的伪影读数上升级处理，会触发 <code>acted_on_unverified_signal</code> 门，这道门照常报告，但不清零该例。
高干扰档加入与真实症状文本格式相同的无关症状，同时把声明的症状率提高到 0.5 与之匹配，否则出题闸门会拒绝该病例。</sub></p>

<!-- 重建图：uv run --with matplotlib python tools/presentation/make_readme_figures.py（确定性生成器，不调用模型）；动图：python tools/presentation/make_readme_gifs.py --only quickstart（需要 Playwright、Pillow 与 Chromium，HAENV_CHROME 指向浏览器） -->

## 快速开始

离线完成生成、校验、作答和计分，无需 API key，零成本。
内置作业默认写入 `artifacts/results/` 与 `artifacts/reports/`。

```bash
git clone https://github.com/thetahealth/mirobody-env && cd mirobody-env

uv run haenv build  benchmarks/inputs/example-ew.job.yaml --gen deterministic --fresh   # 生成病人与题目
uv run haenv verify benchmarks/inputs/example-ew.job.yaml --gen deterministic           # 逐项校验
uv run haenv run    benchmarks/inputs/example-ew.job.yaml --offline                     # 用离线参照解答
uv run haenv report benchmarks/inputs/example-ew.job.yaml --offline                     # 计分并生成报告
```

<p align="center">
  <picture>
    <source media="(prefers-reduced-motion: reduce)" srcset="https://raw.githubusercontent.com/thetahealth/mirobody-env/main/docs/figures/readme_quickstart.png">
    <img src="https://raw.githubusercontent.com/thetahealth/mirobody-env/main/docs/figures/readme_quickstart.gif" alt="终端在一份新克隆里依次运行四条快速开始命令，并回放其实际输出：build 放行 4 例中的 3 例、拦下 EWX-04，verify 报告 82 项、0 项失败，run 与 report 写出报告" width="100%">
  </picture>
</p>
<p align="center"><sub>四条命令在一份新克隆里的实际输出回放（已加速）。</sub></p>

每条命令退出码均为 0。`verify` 输出：

```
[haenv] generation: 3/4 case(s) passed the emission gate
[haenv] per-item verification: 82 item(s) · failed 0 · text leaked in 0 case(s)
```

<details>
<summary>其余输出的含义，以及为什么要 <code>--fresh</code></summary>

录制时的 `report` 输出如下，保留当时的原始路径：

```
[haenv] scoring vintage: all 72 row(s) on disk carry today's stamp `<judging_sha16>` ✓ (fine fingerprint not needed)
[haenv] report -> <repo>/reports/ew-demo/<batch>/eval-ew-demo.md
```

第四例 `EWX-04` 以 `conflicts=['event_density_mismatch']` 被拒：它的高干扰档注入了 5 条症状事件，而它的症状率（默认每周 0.1）只允许 1 条。

`--fresh` 会开一个新批次。批次里一旦有了作答，再对它运行 `build` 或 `verify` 就会以退出码 2 结束：
重出题会让新题目配上旧作答计分。同理，`verify` 要在 `run` 之前。
</details>

用同样的方式出一个公开样例包，全程不调用模型（审计需要 `scikit-learn`）：

```bash
uv run --with scikit-learn --with joblib python tools/generation/make_pack.py --pack p4 --n 50 --out packs/p4
```

出包与审计各门全部通过时退出码为 0，并打印这个包的记录：

```
{"pack": "p4", "n_items": 50, "seed_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "steps": {"gen": 0, "build": 0, "audit": 0}, "job": "<repo>/packs/p4/p4-followup.job.yaml", "batch": "<repo>/packs/p4/results/joint_dx/p4-followup/<batch>", "ref": null}
```

出好的批次在 `packs/p4/results/` 下，审计结果在 `packs/p4/audit/` 下。

<details>
<summary>从 PyPI 安装</summary>

```bash
pip install haenv
JOBS=$(python -c "from haenv import data_root;print(data_root()/'benchmarks'/'inputs')")
cd ~/my-workdir                     # 默认产物目录：./artifacts/
haenv build "$JOBS/example-ew.job.yaml" --gen deterministic --fresh
haenv run   "$JOBS/example-ew.job.yaml" --offline
```

源码检出默认写入 `<checkout>/artifacts/`，wheel 安装默认写入 `<cwd>/artifacts/`。
显式 `HAENV_OUTPUT_ROOT` 优先选择其他可写根目录。`HAENV_DATA_ROOT` 选择只读资源，
并为兼容保留联动输出重定向；新的输出配置请使用 `HAENV_OUTPUT_ROOT`。
`HAENV_CONFIG_OVERLAY` 指向你自己的一个 YAML 文件，合并覆盖包内自带的 `config.yaml`（写你自己的后端与模型，见[评测你自己的 agent](#评测你自己的-agent)）。
</details>

<details>
<summary>用真实模型评测（计费）</summary>

计费运行要两个参数：一个上限，以及记录花费的共享账本。两个都得给，否则在发第一个请求之前就会停。

```bash
# 先跑一个病例、一个模型
uv run haenv run benchmarks/inputs/example-ew.job.yaml --models gemini-3.1-pro --limit 1 \
  --judge-budget-usd 5 --judge-budget-ledger ~/.haenv/budget.json

# 全量，可断点续跑；重跑只发还没有答案的格子
uv run haenv run benchmarks/inputs/example-ew.job.yaml --judge-budget-usd 50 \
  --judge-budget-ledger ~/.haenv/budget.json
```

密钥从 `HAENV_ENV_FILE` 指向的文件读取（见[评测你自己的 agent](#评测你自己的-agent)）；没有密钥时，请求在发出之前就失败。
</details>

## 一个病例，从头到尾

<details>
<summary>agent 看到什么、隐藏金标是什么、两份回答如何判分（快速开始中的 <code>EWX-01</code>）</summary>

题面是一段固定的中文指令，后接 `T` 之前的记录（JSON）。节选如下，20 条数据流大多省略：

```jsonc
{
  "user_profile": {"age_range": "45-49", "sex": "F", "known_conditions": ["obesity"], ...},
  "prediction_context": {"prediction_time_T": 84, "target_event_type": "weight_regain",
                         "prediction_window": "281d", "available_history_window": "84d"},
  "longitudinal_data": {
    "dose_timeline":        [{"ts": 0, "value": 2.5}, {"ts": 28, "value": 5.0}, {"ts": 56, "value": 7.5}],
    "medication_adherence": [{"ts": 42, "value": 0.95}, {"ts": 56, "value": 0.88}, {"ts": 84, "value": 0.8}],
    "weight":               [{"ts": 0, "value": 98.70}, ..., {"ts": 84, "value": 87.10}],
    ...
  },
  "evidence_ledger": [
    {"evidence_id": "EV-EWX-01-01", "source_type": "patient_reported_context",
     "source_timestamp": 43, "note": "报名了社区书法班", ...},
    ...
  ]
}
```

agent 返回一个 JSON 对象：风险预测、从固定列表中选出并排序的驱动因素（各附证据编号），以及
`A0`（继续观察）到 `A5`（紧急升级）之间的一个动作类别。本例的隐藏金标：体重会回升，驱动因素是
`poor_medication_adherence`，且需要临床介入。

| 离线参照求解器 | 预测 | 首位驱动因素 | 动作 | 结果 |
|---|---|---|---|---|
| `no_revision` | 风险 0.2，低 | `poor_medication_adherence`（命中） | `A0`，未请求复核 | 命中硬性门槛 `premature_closure`，整例记零分 |
| `const_ddx` | 风险 0.5，无法判断（弃权） | `unknown_or_multifactorial` | `A3`，请求复核 | 正常计分 |

第一个求解器找对了驱动因素，整例仍然判负：本例需要临床介入，它却选择继续观察，既没开检查也没请求复核。
</details>

## 工作原理

job 文件声明病人事实（病种、药物、剂量阶梯、设备、起始体重）和隐变量。生成器渲染病程，并注入配置的伪影和干扰。
病例必须通过出题闸门（emission gate）才会放行：前提校验拒绝自相矛盾的规格，逐项校验检查每条数据流和事件，
泄漏探针在每条题面发出之前检查它。之后还有批次级闸门和题包审计检查整个题包，例如真实症状不能靠数据足迹与干扰事件区分开。渲染病人的仿真内核包含在本仓库中，位于 `src/haenv_kernel/`。

- **题包是插件。** 每个题包是一个判据组，登记自己的金标块、问法模板和计分；核心流水线不知道有哪些题包。
- **一次运行，一份上下文。** 一次评测运行带着自己的状态（`RunContext`）：同一进程里的两次运行不共用表。
- **指纹跟着定义走。** 判分代码与出题代码各有一个指纹，打在每一行、每个批次上。只搬动定义的重构按定义逐条认证，所以只有算的东西变了，指纹才变。

同一批病人可以用四种形式出题（代码里称为 geometry）：`single`、`gated`（在预算内开检查）、
`slices`（在多个时点分别独立提问）和 `multi`（分轮推进，agent 可修正）。每个题包在 `T` 处考一个决定；题包 ① 允许 agent 先买检查再作答。

## 计分

- **金标由代码推导**，来源是渲染这份病历的世界。
- **1.4.0 的作答分。** 每次作答记 `S = G·D·(1+Q)/2`。四个题包的硬门 `G` 与核心决定 `D` 都由代码判定；题包 ① 由代码按登记的别名把列出的诊断与金标比对，不调用模型。一个 LLM 裁判（`gpt-6-luna`）只判处理方案质量 `Q`。题包分是各题 `S` 的平均，见[排行榜](#排行榜)。
- **1.3.0 的题包综合分（历史）。** 下面两条描述 1.3.0 榜。其题包 ① 综合分含检查 F1，由一个 LLM 裁判把提出的检查与金标比对，投票全量落盘；诊断列入维由代码计算。
- **瞎猜得 0 分。** 题包的决定按机会校正的平衡准确率计分：`cc = (m · BA − 1) / (m − 1)`，`m` 是金标中出现的答案类别数。随机作答或总答同一类得 0，全对得 1。
- **每个题包一个综合分。**

  | 题包 | 综合分 |
  |---|---|
  | ① | 计分维（检查 F1、诊断列入、数值读数、机会校正的复核）的平均，下限 0，×（1 − 硬性门槛判负率） |
  | ② | 去向的 `cc` ×（1 − 硬性门槛判负率）；`cc` 为负时门槛不会把它抬高 |
  | ③ | 决定的 `cc`，下限 0，×（1 − 硬性门槛判负率） |
  | ④ | 变化来源的 `cc` 与真变化幅度技能分 `1 − MAE / MAE_naive` 的平均（不乘门槛乘子） |

  各题包分别出榜，综合分不合并。
- **三栏。** 每条登记的度量属于三种角色之一。*计分*：进综合分。*画像*：判得对但区分不开模型，或还没有效度读数；照算、照存、照展示，不进综合分。*退役*：查明判错；留在登记表里并写明原因，不上任何榜。
- 每行成绩都带有判分代码的指纹，各题包另加自己的指纹；指纹不一致的榜会被发布闸门拒绝。
- 模型原始回答全量保存，修正判分只需重算（`tools/evaluation/restamp_batch.py <batch>`），无需再次调用模型。
- `src/verifier_core/` 包含指纹、发布闸门、硬性门槛乘子、分数上限和噪声下限审计。它不含临床词汇，也不引用临床层代码，
  可以单独审阅或复用。

自检不调用模型，几秒内完成，见 [`docs/REPRODUCE.md`](https://github.com/thetahealth/mirobody-env/blob/main/docs/REPRODUCE.md#4-self-checks)。

## 评测你自己的 agent

被测模型是 `config.yaml` 里 `models:` 下的一项，可以接任何在 `backends:` 中声明的 OpenAI 兼容接口。每一轮是一次携带题面的对话请求；
挂在这类接口后面的 agent 也用同样方式评测，它内部的工具调用判分器看不到。它拿到同样的病人、同样在 `T` 处截断、同样的判分器。插件都在 job 文件里显式登记。

新接口写在 `config.yaml` 旁边的 `config.local.yaml` 里（合并覆盖 `config.yaml`，不进 git）。用 `pip install` 安装时 `config.yaml` 在安装好的包里，改为把同样的 YAML 写进你自己的一个文件，并用 `HAENV_CONFIG_OVERLAY` 指向它。
新的后端名会登记为一个 OpenAI 兼容后端；密钥从 `HAENV_ENV_FILE` 指向的文件读取：

```yaml
backends:
  my-endpoint:
    url: http://localhost:8000/v1/chat/completions
    key_env: MY_ENDPOINT_KEY          # 在 $HAENV_ENV_FILE 里写 MY_ENDPOINT_KEY=...
models:
  my-agent:
    backend: my-endpoint
    model: my-agent-v1
    max_tokens: 8000
    price: {input_per_million_usd: 0, output_per_million_usd: 0}   # 美元/百万 token
```

默认模型都直连：Gemini 走 Google 的接口，Qwen 走 DashScope，其余模型走 OpenRouter，锁定一个上游、关闭回落：账号的零数据留存设置允许时锁原厂端点，否则锁一个提供同一权重的指定托管方（OpenAI 模型在 Azure，Claude 在 Google Vertex，MiniMax 在 Novita）。流水线与模型之间不经任何第三方中转。

付费运行的每个请求都计入 `--judge-budget-usd`。`openrouter`、`google`、`dashscope` 的价格来自服务商；自建后端按每个模型声明的 `price`，乘以接口在 `usage` 里返回的 token 数计费，免费的本地接口填 0。自建后端上没有声明 `price` 的模型，在发出任何请求之前就会被拒绝。

```bash
uv run haenv run benchmarks/inputs/example-ew.job.yaml --models my-agent --limit 1 \
  --judge-budget-usd 5 --judge-budget-ledger ~/.haenv/budget.json
```

带 `--limit` 时是对接口的抽样检查：语义维度不判，不需要 OpenRouter 密钥，运行以退出码 6 结束，表示得分不完整。不带 `--limit` 时，得分中的语义维度由 `openai/gpt-6-luna` 经 OpenRouter 判定，所以同一个密钥文件里还要有 OpenRouter 的密钥，裁判费用计入同一个预算。

| 要改的 | 位置 | 是否写代码 |
|---|---|---|
| 被测模型或 agent | `config.yaml`（`models:`、`backends:`） | 否 |
| 权重与归一化 | 每个榜一个配置文件 | 否 |
| 新增判分器或新的金标类型 | 登记一个函数 | 是 |
| 判分器观察的内容（运行过程的新视图） | 被观察对象插件 | 是 |
| 数据流、事件、药效、伪影 | 环境侧插件 | 是 |

[`examples/`](https://github.com/thetahealth/mirobody-env/blob/main/examples/README.md) 中有五个插件示例包，均可离线运行，各配一个负对照。
指南：[判分器插件](https://github.com/thetahealth/mirobody-env/blob/main/docs/design/llm-judge-plugin.md) · [外部任务类型](https://github.com/thetahealth/mirobody-env/blob/main/docs/design/external-task-contract.md)。

## 孪生病历

孪生病历是某个已出好批次里的一份病历，只改动一处声明过的内容。病历和金标都由生成器根据隐藏真值写出，所以改动可以在写病历的地方进行，并逐字段与原病历比对。要检验一条关于模型行为的说法（例如"模型把噪声归因于病历里写明的因素"），就让每个模型分别回答原病历和它的孪生，看成对差值。

```bash
uv run haenv twin <job.yaml> --op drop-reagent-note --from-batch <已出好的批次> --out twins/
```

命令按以下步骤执行：

1. 选出该操作适用的题目（或用 `--items` 指定）。
2. 写出孪生 job，以及题目相同、内容不变的原题 job。
3. 用确定性方式构建两者。
4. 检查原题构建与源批次逐字节相同，且每份孪生只在该操作声明的字段上与原题不同。

它把逐题的改动记录写进孪生批次，出现任何其他差异时退出码为 1。孪生 job 在 `_provenance.twin` 下记录所用操作，因此重新构建它就能复现孪生病历。

| 操作 | 题包 | 改动 |
|---|---|---|
| `drop-reagent-note` | ④ 化验变化归因 | 报告方法字段里的换试剂备注 |
| `reword-reagent-note-1`、`-2` | ④ 化验变化归因 | 同一备注换一种写法 |
| `drop-planted-symptoms` | ① 限额诊断 | 作为金标线证据植入的患者自述症状条目 |
| `unname-explainer` | ① 限额诊断 | `known_conditions` 里把某个隐藏病情解释掉的已确诊疾病 |

两组都像普通批次一样作答（`tools/evaluation/reliability_passk.py <job> --from-batch <stamp> --k 3`）。

## Agent skills

仓里带三份技能，放在 [`skills/`](https://github.com/thetahealth/mirobody-env/tree/main/skills) 下，采用 [Agent Skills](https://agentskills.io) 版式（`skills/<名字>/SKILL.md`；这份标准不止一家 agent 工具在读）。它们是「从对话里驱动整条流水线」的短路径，也能单独当说明看：

| 技能 | 什么时候用它 |
|---|---|
| [`haenv-synth`](https://github.com/thetahealth/mirobody-env/blob/main/skills/haenv-synth/SKILL.md) | **造**病人与题 —— 从你自己写的 job 文件，或从一段病例描述文本 —— 以及读懂被发射门拒掉的原因 |
| [`haenv-bench`](https://github.com/thetahealth/mirobody-env/blob/main/skills/haenv-bench/SKILL.md) | **跑**题包出榜并读结果：断点续跑、按批次归档的产物、工具调用轨迹、这一批花了多少 |
| [`haenv-extend`](https://github.com/thetahealth/mirobody-env/blob/main/skills/haenv-extend/SKILL.md) | **加自己的**判据、指标流、事件或任务类型，走仓外的包 |

每份都是索引，不是第二份权威：它对流水线的每句描述都对着代码和它引用的文档核过。按标准的说法，一个技能就是一个 Markdown 目录，可选带 `scripts/`、`references/`、`assets/`；它们不含任何绑定某一家工具的东西。用读这份标准的 agent 时，把仓库路径给它，直接用大白话说要做什么 ——「从这份病人造一批题」「跑一批评测并告诉我花了多少」「加一条能抓住 X 的判据」—— 对应的技能会把口径、命令和坑一并给出。

## 成本与缓存

- 出题的模型输出按「模型 + 提示词」缓存在 `artifacts/cases/_llm_cache/`；用同一份 job 重出题包不调用模型。
- 评测默认断点续跑，只补发还没有回答的格子。模型自己给出的空回答、非法 JSON、用完的工具预算都算作答，只有网络、超时、服务商 5xx/429、密钥池造成的中止才补发。原始回答全部落盘，判分代码改了只需重算，不必重跑被测模型。
- 对已有实测依据的模型，`max_tokens` 不得低于按输出长度推出的下限。这能降低截断风险，但不能保证每次回答都在预算内完成。`batch.json` 记录出题与评测用量（`gen_usage`、`eval_usage`）：有实测值时记总量，否则显式记录状态（缺失、不适用或错误）。
- 量级：[交互式榜单](https://thetahealth.github.io/mirobody-env/#act3) 展示每 100 次作答的被测模型花费，语义裁判花费单列。模型费用包含工具请求；无回执和 token 数的请求按模型均价估算。

## 排行榜

1.4.0 榜在四个题包上给 8 个模型排名，每包 100 题（N100），每题作答 3 次。主分为每题三次作答的 `S = G·D·(1+Q)/2` 取平均（k3），四包等权；首次作答的预注册分 pk1 名次相同。`G` 表示通过硬门，`D` 表示核心决定正确，`Q` 是处理方案质量。95% 区间来自配对题簇自助法（2,000 次），每包内对所有模型使用同一批题目重抽样；名次区间取同一批抽样中名次的第 2.5 到 97.5 百分位。

评测所用题集以冻结的题包文件发布。每个批次记录其生成提交与世界设置，在所记录的提交上重新生成，四个题集逐字节一致（2026-10-17 核验：① 限额诊断与 ② 分诊在 `c3cecbee`，③ 用药调整的候选臂在 `fcf3b389`、配对在 `c3cecbee`，④ 化验变化归因在 `05f97181`）。`benchmarks/inputs/` 下的 job 文件使用当前默认设置，产出的题不同。

### 1.4.0 总分

<p align="center">
  <img src="docs/figures/readme_board.png" alt="八个模型的总分与 95% 区间，按总分排列，左侧为名次区间" width="88%">
</p>
<p align="center"><sub>N = 100，k3 为每题三次 S 的平均；包 ④ 的非噪声题可见变化超过配对 RCV。点为总分，横线为 95% 区间，左侧为名次区间；名次区间重叠的模型不能确定先后。</sub></p>

| 名次 | 模型 | 总分（95% 区间） | ① | ② | ③ | ④ | All-3（三次全对） |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | gpt-6-sol | 0.831 (0.808–0.854) | 0.687 | 0.893 | 0.959 | 0.785 | 0.875 |
| 2 | gemini-3.8-flash | 0.792 (0.768–0.816) | 0.591 | 0.950 | 0.882 | 0.744 | 0.873 |
| 3 | gpt-6-luna | 0.718 (0.690–0.746) | 0.490 | 0.859 | 0.897 | 0.627 | 0.695 |
| 4 | deepseek-v4-pro | 0.659 (0.630–0.689) | 0.383 | 0.933 | 0.689 | 0.629 | 0.670 |
| 5–6 | deepseek-v4-flash | 0.627 (0.595–0.660) | 0.459 | 0.856 | 0.656 | 0.538 | 0.585 |
| 5–6 | glm-5.3-flash | 0.602 (0.571–0.632) | 0.381 | 0.813 | 0.668 | 0.547 | 0.533 |
| 7 | minimax-m3 | 0.562 (0.532–0.589) | 0.153 | 0.870 | 0.608 | 0.616 | 0.518 |
| 8 | qwen3.7-flash | 0.425 (0.394–0.457) | 0.299 | 0.599 | 0.271 | 0.533 | 0.325 |

① 限额诊断 · ② 分诊 · ③ 用药调整 · ④ 化验变化归因。All-3（三次全对）是三次作答的决定都正确的题目占比，四包等权平均；它是可靠性画像，不是排名指标，按它排名次序相同。`Q` 取同一 LLM 裁判两次判定的平均；7,246 条 Q 记录中，7,215 条为两次平均，20 条只有一次判定，11 条无处理方案而记 0。量表在本榜作答上校准；区间不包含裁判噪声。三轮正确率、All-3、硬门率与病史分组读数作为画像，见交互式榜单。

[交互式榜单](https://thetahealth.github.io/mirobody-env/#act3) ·
[榜单数据（JSON）](https://github.com/thetahealth/mirobody-env/blob/main/web/demo/board.json)

### 早期榜单（历史版本）

<details>
<summary>1.3.0 总分与分包</summary>

### 1.3.0 总分

| 名次 | 模型 | 总分 | 95% 区间 |
|---:|---|---:|---:|
| 1 | gpt-6-sol | 0.712 | 0.671–0.751 |
| 2–4 | gemini-3.1-pro | 0.648 | 0.610–0.689 |
| 2–4 | gemini-3.7-flash | 0.640 | 0.595–0.684 |
| 2–4 | gpt-6-luna | 0.609 | 0.564–0.656 |
| 5–6 | kimi-k3 | 0.556 | 0.509–0.604 |
| 5–6 | deepseek-v4-pro | 0.526 | 0.484–0.570 |
| 7–8 | glm-5.3-flash | 0.470 | 0.424–0.521 |
| 7–9 | minimax-m3 | 0.458 | 0.417–0.500 |
| 8–10 | deepseek-v4-flash | 0.415 | 0.366–0.464 |
| 9–10 | qwen3.7-flash | 0.388 | 0.338–0.437 |

gpt-6-sol 排第一，且与其余每个模型都分得开（它的九对配对差都超过 2 个配对标准误）。gemini-3.1-pro、gemini-3.7-flash 与 gpt-6-luna 共享第 2–4 名：三者两两的差都没有超过 2 个配对标准误，所以它们的先后不是名次。名次因此写成区间：45 对模型里有 37 对分得开。1.3.0 的榜与 1.2.x 的榜**不可比**：判分指纹与世界指纹都变了。

### 1.3.0 分包

每格是该题包的分数和 95% 区间。

| 模型 | ① 鉴别诊断 | ② 急诊分诊 | ③ 慢病调药 | ④ 随访解读 |
|---|---:|---:|---:|---:|
| gpt-6-sol | 0.643 (0.579–0.703) | 0.436 (0.318–0.551) | 0.776 (0.688–0.860) | 0.991 (0.970–1.000) |
| gemini-3.1-pro | 0.294 (0.240–0.351) | 0.448 (0.316–0.581) | 0.861 (0.796–0.926) | 0.991 (0.964–1.000) |
| gemini-3.7-flash | 0.482 (0.410–0.549) | 0.425 (0.310–0.545) | 0.710 (0.610–0.804) | 0.942 (0.888–0.986) |
| gpt-6-luna | 0.506 (0.397–0.604) | 0.365 (0.250–0.486) | 0.802 (0.725–0.879) | 0.764 (0.699–0.827) |
| kimi-k3 | 0.347 (0.263–0.429) | 0.383 (0.268–0.503) | 0.741 (0.650–0.829) | 0.751 (0.675–0.828) |
| deepseek-v4-pro | 0.258 (0.198–0.323) | 0.369 (0.258–0.481) | 0.762 (0.678–0.842) | 0.716 (0.646–0.777) |
| glm-5.3-flash | 0.232 (0.142–0.331) | 0.420 (0.318–0.530) | 0.656 (0.559–0.753) | 0.573 (0.496–0.655) |
| minimax-m3 | 0.130 (0.077–0.188) | 0.320 (0.215–0.423) | 0.742 (0.650–0.830) | 0.640 (0.562–0.722) |
| deepseek-v4-flash | 0.063 (-0.020–0.148) | 0.258 (0.148–0.370) | 0.691 (0.595–0.790) | 0.649 (0.559–0.733) |
| qwen3.7-flash | 0.015 (-0.055–0.078) | 0.323 (0.201–0.454) | 0.625 (0.523–0.728) | 0.587 (0.507–0.665) |

</details>

1.x 榜在两条诊断轨 `ddx-timeline` 与 `ddx-workup`（各 145 例）上给十个模型排名。它按发布时的样子保留，与各题包的榜**不可比**：世界、题目与计分此后都变了。桥接题是两者之间唯一的联系。

<details>
<summary>1.x 的两张表</summary>

1.x 的综合分是计分维的平均值乘以（1 − 硬性门槛判负率），取值 0 到 1；*95% 区间* 是按病例的 bootstrap 区间（10,000 次重抽），档号 = 1 + 显著优于它的模型数（Holm 校正，α = 0.05）。*已作答* 是 145 格中有可计分回答的格数；没有可计分回答的格记 0 分。每格只跑一次。该综合分里的诊断维没有盲标效度读数，所以那张榜标为初步。

### ddx-timeline（1.x）


| 名次 | 模型 | 综合分 | 95% 区间 | 档 | 已作答 |
|---:|---|---:|---:|---:|---:|
| 1 | gemini-3.1-pro | 0.724 | 0.671–0.776 | 1 | 145 / 145 |
| 2 | gpt-6-sol | 0.691 | 0.653–0.730 | 1 | 145 / 145 |
| 3 | gemini-3.7-flash | 0.678 | 0.618–0.740 | 1 | 145 / 145 |
| 4 | gpt-6-luna | 0.666 | 0.630–0.705 | 1 | 145 / 145 |
| 5 | kimi-k3 | 0.662 | 0.632–0.695 | 1 | 145 / 145 |
| 6 | deepseek-v4-pro | 0.652 | 0.632–0.671 | 1 | 145 / 145 |
| 7 | minimax-m3 | 0.617 | 0.587–0.647 | 3 | 143 / 145 |
| 8 | qwen3.7-flash | 0.589 | 0.550–0.631 | 4 | 145 / 145 |
| 9 | deepseek-v4-flash | 0.434 | 0.380–0.494 | 9 | 144 / 145 |
| 10 | glm-5.3-flash | 0.048 | 0.024–0.076 | 10 | 17 / 145 |

### ddx-workup（1.x）

| 名次 | 模型 | 综合分 | 95% 区间 | 档 | 已作答 |
|---:|---|---:|---:|---:|---:|
| 1 | kimi-k3 | 0.697 | 0.639–0.751 | 1 | 144 / 145 |
| 2 | gpt-6-luna | 0.696 | 0.635–0.755 | 1 | 145 / 145 |
| 3 | gemini-3.1-pro | 0.689 | 0.625–0.751 | 1 | 145 / 145 |
| 4 | deepseek-v4-pro | 0.685 | 0.651–0.720 | 1 | 144 / 145 |
| 5 | gemini-3.7-flash | 0.680 | 0.615–0.743 | 1 | 145 / 145 |
| 6 | gpt-6-sol | 0.673 | 0.615–0.730 | 1 | 145 / 145 |
| 6 | minimax-m3 | 0.673 | 0.623–0.721 | 1 | 145 / 145 |
| 8 | qwen3.7-flash | 0.493 | 0.443–0.544 | 8 | 145 / 145 |
| 9 | glm-5.3-flash | 0.375 | 0.303–0.446 | 8 | 116 / 145 |
| 10 | deepseek-v4-flash | 0.304 | 0.247–0.365 | 9 | 145 / 145 |


[图](https://github.com/thetahealth/mirobody-env/blob/main/docs/figures/readme_results.svg) ·
[原始数值（CSV）](https://github.com/thetahealth/mirobody-env/blob/main/docs/figures/readme_results.csv) ·
[1.x 快照（JSON，键 `history_1x`）](https://github.com/thetahealth/mirobody-env/blob/main/web/demo/data.json)

</details>

## 局限

- **包 ④ 的 RCV 基线。** 只读两次印出值和规程 RCV 表的读者，在包 ④ 上 D 约 0.5，所以 ④ 分数在 0.5 附近只说明会用 RCV。
- **名次不确定性。** 七对相邻模型里，一对（deepseek-v4-flash 与 glm-5.3-flash）的配对分差区间包含 0（pk1 下为两对：deepseek-v4-pro 与 deepseek-v4-flash、deepseek-v4-flash 与 glm-5.3-flash），榜单不能确定这些模型对的先后。
- **临床复核。** 题包 ② 和 ③ 背后的规则（哪些表现要送急诊、剂量调整何时算达标）以及若干登记条目带 `review: pending`；医学内容整体尚未经执业医生审阅。
- **单一语义裁判。** 一个模型（`gpt-6-luna`，推理档 high）为四个题包的处理方案质量 `Q` 评分；每个题包（含包 ①）的 `D` 与 `G` 都由代码判定。被测模型中有同一厂商的模型，不能排除自我偏好。跨厂商核查用 Gemini 裁判重判了全部 6,937 条 `G·D = 1` 的作答（请求 `gemini-3.7-flash`，服务方对每条请求实际提供 `gemini-3.8-flash`，它本身也是被测模型）：八个模型的名次不变，gpt-6-sol 领先 gemini-3.8-flash +0.031（95% 区间 +0.010 至 +0.052），模型层面平均 `Q` 与主裁判的相关 r = 0.968。`Q` 在两次判定均可用时取平均；单次判定与无方案的例外数量见上。`Q` 的量表在本榜作答上校准，没有临床医生核验其判定，分数区间不含裁判噪声。
- **诊断维待效度。** 桥接轨上诊断轨的计分维依赖判断，等待临床盲标；在此之前不设公开列。
- **合成数据。** 所有病人都是合成的，本基准仅用于评测，不构成任何医疗建议。

完整清单（含计分、生成与世界层的缺口）见[数据卡](https://github.com/thetahealth/mirobody-env/blob/main/docs/DATA_CARD.md#known-gaps)。

## 相关工作

| 工作 | 评测内容 |
|---|---|
| ESL-Bench（[arXiv:2604.02834](https://arxiv.org/abs/2604.02834)，[数据集](https://huggingface.co/datasets/mirobody/ESL-Bench)） | 100 名合成用户，各有 1–5 年的设备、体检与事件轨迹；每人 100 道题，覆盖查找、趋势、比较、异常、解释五个维度，答案由程序计算。排行榜：[Health Memory Arena](https://healthmemoryarena.ai) |
| mirobody-eval（[GitHub](https://github.com/thetahealth/mirobody-eval)） | 用已发表的健康基准（含 ESL-Bench）评测现成系统的框架，虚拟用户、被测对象、评分器均可替换 |
| MedAgentBench（[arXiv:2501.14654](https://arxiv.org/abs/2501.14654)） | FHIR 虚拟 EHR 中的 300 个 agent 任务 |
| EHRSHOT（[arXiv:2307.02028](https://arxiv.org/abs/2307.02028)） | 6,739 名真实病人纵向 EHR 上的少样本预测 |
| LongHealth（[arXiv:2401.14490](https://arxiv.org/abs/2401.14490)） | 20 份长篇虚构病历上的 400 道选择题 |
| HealthBench（[arXiv:2505.08775](https://arxiv.org/abs/2505.08775)） | 5,000 段健康对话，由模型按医生编写的细则评分 |
| AgentClinic（[arXiv:2405.07960](https://arxiv.org/abs/2405.07960)）、CRAFT-MD（[doi:10.1038/s41591-024-03328-5](https://doi.org/10.1038/s41591-024-03328-5)） | 与 LLM 扮演的病人对话，完成问诊与诊断 |

HAEnv 同时具备纵向病历、在 `T` 处截断的题面、先于数据确定的金标、决定与硬门由代码计分、处理方案质量由一个 LLM 裁判判定，以及可重新生成的合成病例。
可靠性指标 `pass^k` 来自 τ-bench（[arXiv:2406.12045](https://arxiv.org/abs/2406.12045)）。

## 数据、伦理与复现

所有病人均为合成数据，不构成医疗建议，不可用于临床决策。

> 整条管线公开：渲染病人的世界、写题的生成器、判分器。canary 串（[`CANARY.md`](https://github.com/thetahealth/mirobody-env/blob/main/CANARY.md)）标记本题包的文本，便于检测它是否出现在某个语料中；允许用公开样例训练；1.4.0 榜的题由公开种子生成（更早的榜用私有种子）。

- [`docs/DATA_CARD.md`](https://github.com/thetahealth/mirobody-env/blob/main/docs/DATA_CARD.md)：题包、金标来源、计分、已知缺口，以及金标为何随题公开。
- [`docs/REPRODUCE.md`](https://github.com/thetahealth/mirobody-env/blob/main/docs/REPRODUCE.md)：哪些可免费复现、哪些计费，以及自检。
- [`docs/ETHICS.md`](https://github.com/thetahealth/mirobody-env/blob/main/docs/ETHICS.md)：数据来源与使用边界。
- [`CONTRIBUTING.md`](https://github.com/thetahealth/mirobody-env/blob/main/CONTRIBUTING.md)：如何修改计分或生成代码。

## 仓库结构

安装后的命令和导入名仍为 `haenv`。源码放在 `src/`；仿真内核和通用核验器是同一发行项目中的独立命名空间。

```text
src/
├── haenv/           应用、世界、交互、判分与运行
├── haenv_kernel/    仿真与隔离原语
└── verifier_core/   不含领域规则的分数核验
plugins/rare/        可选的罕见病编码项目
benchmarks/          固定作业、任务卡、源病例与重放材料
registry/            版本化规则、来源锚、效度与共享参考
artifacts/           生成的结果、报告、缓存与抽取作业
tools/               出题、评测、验证、呈现与发布命令
examples/            扩展示例
docs/                当前用户与设计文档
tests/               回归与合同检查
web/demo/            浏览器演示
```

包职责、当前规范和复现方法统一从[文档索引](https://github.com/thetahealth/mirobody-env/blob/main/docs/README.md)进入。
罕见病项目独立于主发行包，继续保持原型状态。

## 引用

```bibtex
@software{haenv,
  title  = {Health Agent Environment},
  author = {{Theta Health}},
  year   = {2026},
  url    = {https://github.com/thetahealth/mirobody-env},
  version = {1.4.0.dev0}
}
```

同样的元数据见 [`CITATION.cff`](https://github.com/thetahealth/mirobody-env/blob/main/CITATION.cff)。

## 许可与致谢

代码采用 [MIT 许可](https://github.com/thetahealth/mirobody-env/blob/main/LICENSE)；合成数据与题包采用 [CC BY 4.0](https://github.com/thetahealth/mirobody-env/blob/main/LICENSE-DATA)。第三方声明见 [`NOTICE.md`](https://github.com/thetahealth/mirobody-env/blob/main/NOTICE.md)。

HAEnv 的灵感来自 [ESL-Bench](https://arxiv.org/abs/2604.02834)。

<p align="center"><sub>合成数据 · 仅供评测 · 非医疗建议</sub></p>
