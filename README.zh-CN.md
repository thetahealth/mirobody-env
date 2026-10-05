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

> 整条管线公开：渲染病人的世界、写题的生成器、判分器。canary 串（[`CANARY.md`](https://github.com/thetahealth/mirobody-env/blob/main/CANARY.md)）标记本题包的文本，便于检测它是否出现在某个语料中；允许用公开样例训练；正式榜使用私有种子。

<p align="center">
  <img src="https://raw.githubusercontent.com/thetahealth/mirobody-env/main/docs/figures/readme_patient.png" alt="一个合成病人：2 型糖尿病，使用替尔泊肽。家用秤与诊室秤体重、步数与静息心率、剂量与依从性、上报事件；T 之后的记录以灰底标为隐藏" width="92%">
</p>
<p align="center"><sub>一个合成病人（图中标注为英文）。<code>T</code> 左侧是题面，右侧对 agent 隐藏、用于判分。
依从性在 <code>T</code> 之前下降，由此引起的体重回升在 <code>T</code> 之后、隐藏的反转点开始出现。</sub></p>

## 测什么

HAEnv（Health Agent Environment）的代码仓库是 `mirobody-env`，安装后的 Python 包与命令行工具名为 `haenv`。

HAEnv 评测在随时间展开的病历上做出的临床决定。合成病人的病历按月增长，agent 看到截止时点 `T` 之前的记录，做出一个决定。正确答案由渲染这份病历的世界在写出病历之前就已确定，对 agent 隐藏。

HAEnv 负责生成病人、设定难度、推出金标。要用已发表的健康基准（如 ESL-Bench）评测现成系统，请用
[mirobody-eval](https://github.com/thetahealth/mirobody-eval)。

### 四个题包

基准由四个任务题包组成，每道题考一个核心决定。

| 题包 | `T` 处的决定 | 答案 | 判分方式 |
|---|---|---|---|
| ① 鉴别诊断 | 哪些病能解释这份病历——第二种病藏在被第一种病解释掉的表现背后；是否需要临床复核 | 鉴别诊断、要开的检查、复核标记 | 代码对世界真值判；由一个 LLM 裁判把自由文本的诊断与检查对到金标 |
| ② 急性分诊 | 病人现在该去哪 | `ed_now` · `within_24h` · `routine_followup` · `watchful_waiting` | 代码对世界真值判 |
| ③ 慢病调药 | 这次随访时药该怎么办 | `uptitrate` · `downtitrate` · `maintain` · `switch` · `check_adherence_or_adverse_effect` | 代码对世界记录的剂量线、取药与控制目标判 |
| ④ 随访解读 | 与上次结果相比的变化是不是真的 | `true_change` · `analytic_biological_noise` · `preanalytical` · `method_difference`，以及真变化的幅度 | 代码对世界的真值与逐读数扰动记录判 |

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
uv run --with scikit-learn --with joblib python tools/make_pack.py --pack p4 --n 50 --out packs/p4
```

`--pack` 取 `m2`（①）、`pack2`（②）、`p3`（③）或 `p4`（④）；`--n` 是题量（50、100……）；`--seed`（或 `HAENV_PACK_SEED`）设种子。同一题包、同一题量、同一种子出的包相同。job 带 `pack: {n_items, seed_sha256}` 头，`batch.json` 记录题量、种子的哈希和 job，从不记种子本身。任何题量下各类占比都按该题包的比例表分配。

| 题包 | job 文件 | N = 50 时的题数 | N = 50 时的答案类别 |
|---|---|---:|---|
| ① 鉴别诊断 | `inputs/m2-pack1.job.yaml` | 50 | 34 道新题（24 道有隐藏病、10 道没有）与 16 道桥接题 |
| ② 急性分诊 | `inputs/pack2-triage.job.yaml` | 50 | `ed_now` 13 · `within_24h` 13 · `routine_followup` 12 · `watchful_waiting` 12 |
| ③ 慢病调药 | `inputs/p3-meds.job.yaml` | 50 | `maintain` 12 · `downtitrate` 12 · `check_adherence_or_adverse_effect` 12 · `uptitrate` 8 · `switch` 6 |
| ④ 随访解读 | `inputs/p4-followup.job.yaml` | 50 | `true_change` 13 · `analytic_biological_noise` 13 · `preanalytical` 12 · `method_difference` 12 |

- **公开样例包。** 随仓发布的 job 文件用公开种子；照着出就得到公开样例包。
- **正式榜用私有种子。** 榜的批次只记种子的哈希，从仓库无法重出榜上的题。换榜时公开旧榜的种子，旧榜随之可复现。
- **审计是发布闸门。** `tools/pack_audit.py` 检查出好的批次：job 计划的每个格子都在，从记录重新推出的金标与存下的金标一致，不看题面的桩与表面特征预测不了答案。审计没过，`haenv run` 拒绝评测该包的题。

桥接轨和两个较小的示例任务以普通 job 文件发布：

| 任务 | job 文件 | 病例规格数 | 形式 |
|---|---|---|---|
| 多时点鉴别诊断（桥接） | `inputs/ddx-timeline.job.yaml` | 145 | 在多个时点分别提问 |
| 限预算开检查（桥接） | `inputs/ddx-workup.job.yaml` | 145 | agent 在预算内自行开检查 |
| 体重回升预测与驱动因素归因 | `inputs/early_warning-20.job.yaml` | 20 | 在 `T` 处提问一次 |
| 多轮随访复盘 | `inputs/tracking_review-20.job.yaml` | 20 | 分轮推进，agent 可修正 |

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

<!-- 重建图：uv run --with matplotlib python docs/scripts/make_readme_figures.py（确定性生成器，不调用模型）；动图：python docs/scripts/make_readme_gifs.py --only quickstart（需要 Playwright、Pillow 与 Chromium，HAENV_CHROME 指向浏览器） -->

## 快速开始

离线完成生成、校验、作答和计分，无需 API key，零成本。

```bash
git clone https://github.com/thetahealth/mirobody-env && cd mirobody-env

uv run haenv build  inputs/example-ew.job.yaml --gen deterministic --fresh   # 生成病人与题目
uv run haenv verify inputs/example-ew.job.yaml --gen deterministic           # 逐项校验
uv run haenv run    inputs/example-ew.job.yaml --offline                     # 用离线参照解答
uv run haenv report inputs/example-ew.job.yaml --offline                     # 计分并生成报告
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

`report` 输出：

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
uv run --with scikit-learn --with joblib python tools/make_pack.py --pack p4 --n 50 --out packs/p4
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
JOBS=$(python -c "import haenv,pathlib;print(pathlib.Path(haenv.__file__).parent/'_data'/'inputs')")
cd ~/my-workdir                     # 产物写到当前目录
haenv build "$JOBS/example-ew.job.yaml" --gen deterministic --fresh
haenv run   "$JOBS/example-ew.job.yaml" --offline
```

`HAENV_DATA_ROOT` 设只读资源根目录，`HAENV_OUTPUT_ROOT` 设产物根目录。
`HAENV_CONFIG_OVERLAY` 指向你自己的一个 YAML 文件，合并覆盖包内自带的 `config.yaml`（写你自己的后端与模型，见[评测你自己的 agent](#评测你自己的-agent)）。
</details>

<details>
<summary>用真实模型评测（计费）</summary>

计费运行要两个参数：一个上限，以及记录花费的共享账本。两个都得给，否则在发第一个请求之前就会停。

```bash
# 先跑一个病例、一个模型
uv run haenv run inputs/example-ew.job.yaml --models gemini-3.1-pro --limit 1 \
  --judge-budget-usd 5 --judge-budget-ledger ~/.haenv/budget.json

# 全量，可断点续跑；重跑只发还没有答案的格子
uv run haenv run inputs/example-ew.job.yaml --judge-budget-usd 50 \
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

<p align="center">
  <a href="https://github.com/thetahealth/mirobody-env/blob/main/docs/figures/readme_how_it_works.png"><img src="https://raw.githubusercontent.com/thetahealth/mirobody-env/main/docs/figures/readme_how_it_works.png" alt="HAEnv 合成评测流程：病人事实与隐变量生成并校验纵向病历；agent 仅接收截至 T 的记录，代码派生的隐藏金标只用于回答评分；未通过出题检查的病例不放行。" width="100%"></a>
</p>
<p align="center"><sub>流程示意：agent 只接收截至 <code>T</code> 的记录，隐藏金标只交给判分器；判分器是代码，另有一个 LLM 裁判判定语义维度（见<a href="#计分">计分</a>）。点击图片可查看原尺寸。</sub></p>

job 文件声明病人事实（病种、药物、剂量阶梯、设备、起始体重）和隐变量。生成器渲染病程，并注入配置的伪影和干扰。
病例必须通过出题闸门（emission gate）才会放行：前提校验拒绝自相矛盾的规格，逐项校验检查每条数据流和事件，
泄漏探针在每条题面发出之前检查它。之后还有批次级闸门和题包审计检查整个题包，例如真实症状不能靠数据足迹与干扰事件区分开。渲染病人的仿真内核包含在本仓库中，位于 `haenv_kernel/`。

- **题包是插件。** 每个题包是一个判据组，登记自己的金标块、问法模板和计分；核心流水线不知道有哪些题包。
- **一次运行，一份上下文。** 一次评测运行带着自己的状态（`RunContext`）：同一进程里的两次运行不共用表。
- **指纹跟着定义走。** 判分代码与出题代码各有一个指纹，打在每一行、每个批次上。只搬动定义的重构按定义逐条认证，所以只有算的东西变了，指纹才变。

同一批病人可以用四种形式出题（代码里称为 geometry）：`single`、`gated`（在预算内开检查）、
`slices`（在多个时点分别独立提问）和 `multi`（分轮推进，agent 可修正）。每个题包在 `T` 处考一个决定；题包 ① 允许 agent 先买检查再作答。

## 计分

- **金标由代码推导**，来源是渲染这份病历的世界。题包 ②③④ 只用代码判分。题包 ① 用代码判分，另由一个 LLM 裁判（`gpt-6-luna`）把自由文本的诊断与检查对到金标，投票全量落盘。
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
- 模型原始回答全量保存，修正判分只需重算（`tools/restamp_batch.py <batch>`），无需再次调用模型。
- `verifier_core/` 包含指纹、发布闸门、硬性门槛乘子、分数上限和噪声下限审计。它不含临床词汇，也不引用临床层代码，
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
uv run haenv run inputs/example-ew.job.yaml --models my-agent --limit 1 \
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

## Agent skills

仓里带三份技能，放在 [`skills/`](https://github.com/thetahealth/mirobody-env/tree/main/skills) 下，采用 [Agent Skills](https://agentskills.io) 版式（`skills/<名字>/SKILL.md`；这份标准不止一家 agent 工具在读）。它们是「从对话里驱动整条流水线」的短路径，也能单独当说明看：

| 技能 | 什么时候用它 |
|---|---|
| [`haenv-synth`](https://github.com/thetahealth/mirobody-env/blob/main/skills/haenv-synth/SKILL.md) | **造**病人与题 —— 从你自己写的 job 文件，或从一段病例描述文本 —— 以及读懂被发射门拒掉的原因 |
| [`haenv-bench`](https://github.com/thetahealth/mirobody-env/blob/main/skills/haenv-bench/SKILL.md) | **跑**题包出榜并读结果：断点续跑、按批次归档的产物、工具调用轨迹、这一批花了多少 |
| [`haenv-extend`](https://github.com/thetahealth/mirobody-env/blob/main/skills/haenv-extend/SKILL.md) | **加自己的**判据、指标流、事件或任务类型，走仓外的包 |

每份都是索引，不是第二份权威：它对流水线的每句描述都对着代码和它引用的文档核过。按标准的说法，一个技能就是一个 Markdown 目录，可选带 `scripts/`、`references/`、`assets/`；它们不含任何绑定某一家工具的东西。用读这份标准的 agent 时，把仓库路径给它，直接用大白话说要做什么 ——「从这份病人造一批题」「跑一批评测并告诉我花了多少」「加一条能抓住 X 的判据」—— 对应的技能会把口径、命令和坑一并给出。

## 成本与缓存

- 出题的模型输出按「模型 + 提示词」缓存在 `cases/_llm_cache/`；用同一份 job 重出题包不调用模型。
- 评测默认断点续跑，只补发还没有回答的格子。模型自己给出的空回答、非法 JSON、用完的工具预算都算作答，只有网络、超时、服务商 5xx/429、密钥池造成的中止才补发。原始回答全部落盘，判分代码改了只需重算，不必重跑被测模型。
- 对已有实测依据的模型，`max_tokens` 不得低于按输出长度推出的下限。这能降低截断风险，但不能保证每次回答都在预算内完成。`batch.json` 记录出题与评测用量（`gen_usage`、`eval_usage`）：有实测值时记总量，否则显式记录状态（缺失、不适用或错误）。
- 量级：题包的一道题是一次请求。N = 50 时跑一遍的费用（按每条题面的 token 数与拟合的服务商单价估算）：题包 ② 每个模型 $0.09–$9.55（十个 1.x 模型平均 $3.20），题包 ③ $0.09–$8.19（平均 $2.65），题包 ④ $0.09–$8.62（平均 $2.82）；题包 ②–④ 没有裁判费用。 细节与重算命令见 [`docs/REPRODUCE.md`](https://github.com/thetahealth/mirobody-env/blob/main/docs/REPRODUCE.md#tokens-and-caching)。

## 排行榜

1.2.0 榜在四个题包上给 10 个模型排名，每包 50 题，每题作答 3 次。总分是四个题包分数的等权平均，各分数取值 0 到 1。95% 区间来自按病例 × 轮次的 bootstrap（4,000 次，题目与轮次一起重抽）；名次按未取整的分数排。榜单题包用私有种子抽取，job 与批次里只记它的 sha256：`8f90c70f61fe7a53860e9a21576c0393e26515ba7eede0467a534360dfa7f9b9`。

### 1.2.0 总分

| 名次 | 模型 | 总分 | 95% 区间 |
|---:|---|---:|---:|
| 1 | gemini-3.1-pro | 0.732 | 0.674–0.788 |
| 2 | gemini-3.7-flash | 0.677 | 0.602–0.742 |
| 3 | gpt-6-sol | 0.671 | 0.617–0.729 |
| 4 | deepseek-v4-pro | 0.560 | 0.484–0.634 |
| 5 | gpt-6-luna | 0.547 | 0.470–0.616 |
| 6 | kimi-k3 | 0.547 | 0.470–0.622 |
| 7 | minimax-m3 | 0.521 | 0.439–0.600 |
| 8 | glm-5.3-flash | 0.483 | 0.408–0.564 |
| 9 | deepseek-v4-flash | 0.426 | 0.347–0.500 |
| 10 | qwen3.7-flash | 0.389 | 0.317–0.465 |

### 1.2.0 分包

每格是该题包的分数和 95% 区间。

| 模型 | ① 鉴别诊断 | ② 急诊分诊 | ③ 慢病调药 | ④ 随访解读 |
|---|---:|---:|---:|---:|
| gemini-3.1-pro | 0.834 (0.746–0.908) | 0.504 (0.364–0.645) | 0.799 (0.684–0.902) | 0.790 (0.691–0.871) |
| gemini-3.7-flash | 0.755 (0.646–0.854) | 0.532 (0.389–0.673) | 0.667 (0.518–0.801) | 0.754 (0.581–0.890) |
| gpt-6-sol | 0.607 (0.539–0.697) | 0.408 (0.265–0.554) | 0.785 (0.662–0.894) | 0.886 (0.815–0.944) |
| deepseek-v4-pro | 0.566 (0.490–0.664) | 0.457 (0.311–0.598) | 0.677 (0.537–0.804) | 0.538 (0.366–0.693) |
| gpt-6-luna | 0.547 (0.485–0.608) | 0.362 (0.234–0.493) | 0.743 (0.579–0.879) | 0.536 (0.339–0.712) |
| kimi-k3 | 0.612 (0.517–0.726) | 0.468 (0.319–0.615) | 0.694 (0.556–0.818) | 0.411 (0.205–0.609) |
| minimax-m3 | 0.552 (0.486–0.632) | 0.421 (0.277–0.569) | 0.806 (0.673–0.919) | 0.305 (0.099–0.491) |
| glm-5.3-flash | 0.594 (0.489–0.717) | 0.497 (0.356–0.631) | 0.566 (0.421–0.703) | 0.277 (0.068–0.476) |
| deepseek-v4-flash | 0.406 (0.292–0.523) | 0.306 (0.175–0.441) | 0.589 (0.460–0.710) | 0.403 (0.203–0.581) |
| qwen3.7-flash | 0.428 (0.341–0.533) | 0.329 (0.187–0.487) | 0.542 (0.413–0.655) | 0.258 (0.050–0.464) |

[交互式榜单](https://thetahealth.github.io/mirobody-env/#act3) ·
[榜单数据（JSON）](https://github.com/thetahealth/mirobody-env/blob/main/web/demo/board.json)

### 1.x 榜（历史版本）

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

- **题包 ② 分不开前几名。** 题包 ② 前六名两两之间没有一对的差超过 1.96 个标准误（15 对中 0 对）；题包 ③ 分开 3 对，题包 ① 8 对，题包 ④ 10 对。总榜前六名 15 对中 9 对可分。
- **临床复核。** 题包 ② 和 ③ 背后的规则（哪些表现要送急诊、剂量调整何时算达标）以及若干登记条目带 `review: pending`；医学内容整体尚未经执业医生审阅。
- **题包 ① 只有一个裁判。** 由一个模型（`gpt-6-luna`，reasoning high）把自由文本的诊断与检查对到金标，投两票、不一致时投第三票。同一厂商的模型也在被测之列，不能排除自我偏好；也没有第二个裁判或医生标注来核对它的判定。题包 ②–④ 没有 LLM 裁判。
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

HAEnv 同时具备纵向病历、在 `T` 处截断的题面、先于数据确定的金标、除语义维度（由一个 LLM 裁判判定）外全部由代码计分，以及可重新生成的合成病例。
可靠性指标 `pass^k` 来自 τ-bench（[arXiv:2406.12045](https://arxiv.org/abs/2406.12045)）。

## 数据、伦理与复现

所有病人均为合成数据，不构成医疗建议，不可用于临床决策。

- [`docs/DATA_CARD.md`](https://github.com/thetahealth/mirobody-env/blob/main/docs/DATA_CARD.md)：题包、金标来源、计分、已知缺口，以及金标为何随题公开。
- [`docs/REPRODUCE.md`](https://github.com/thetahealth/mirobody-env/blob/main/docs/REPRODUCE.md)：哪些可免费复现、哪些计费，以及自检。
- [`docs/ETHICS.md`](https://github.com/thetahealth/mirobody-env/blob/main/docs/ETHICS.md)：数据来源与使用边界。
- [`CONTRIBUTING.md`](https://github.com/thetahealth/mirobody-env/blob/main/CONTRIBUTING.md)：如何修改计分或生成代码。

## 引用

```bibtex
@software{haenv,
  title  = {Health Agent Environment},
  author = {{Theta Health}},
  year   = {2026},
  url    = {https://github.com/thetahealth/mirobody-env},
  version = {1.2.1}
}
```

同样的元数据见 [`CITATION.cff`](https://github.com/thetahealth/mirobody-env/blob/main/CITATION.cff)。

## 许可与致谢

代码采用 [MIT 许可](https://github.com/thetahealth/mirobody-env/blob/main/LICENSE)；合成数据与题包采用 [CC BY 4.0](https://github.com/thetahealth/mirobody-env/blob/main/LICENSE-DATA)。第三方声明见 [`NOTICE.md`](https://github.com/thetahealth/mirobody-env/blob/main/NOTICE.md)。

HAEnv 的灵感来自 [ESL-Bench](https://arxiv.org/abs/2604.02834)。

<p align="center"><sub>合成数据 · 仅供评测 · 非医疗建议</sub></p>
