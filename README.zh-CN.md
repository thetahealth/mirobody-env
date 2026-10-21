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
  <a href="#工作原理">工作原理</a> &middot;
  <a href="#四个题包">题包</a> &middot;
  <a href="#排行榜">评测结果</a> &middot;
  <a href="#快速开始">快速开始</a> &middot;
  <a href="#评测你自己的-agent">评测你的 agent</a> &middot;
  <a href="#文档">文档</a> &middot;
  <a href="https://thetahealth.github.io/mirobody-env/">在线演示</a> &middot;
  <a href="#引用">引用</a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/thetahealth/mirobody-env/main/docs/figures/readme_meds_patient.png" alt="用药调整题包中的一个合成病人，从就诊前 400 天到就诊时点 T：病历印出的 HbA1c 对照隐藏真值与本院目标，每日静息心率与每周家庭收缩压，格列齐特剂量与两次 90 天取药及隐藏的依从性，以及按隐藏病因着色的症状与其他药物调整" width="100%">
</p>
<p align="center"><sub>公开种子用药题包中的一道题，截至就诊时点 <code>T</code>（图中标注为英文）；空心点、依从性和症状颜色对 agent 隐藏。格列齐特已在最高剂量而 HbA1c 仍高于目标，但上次取药后 19 天只服了 12 天的量，所以金标是核查依从性，而不是加量。</sub></p>

## 工作原理

HAEnv 的每道题都从一个隐藏的病人状态出发。**定义：** job 文件设定病人事实（病种、药物、剂量阶梯、设备）和依从性、化验真值等隐变量。**渲染：** 生成器据此写出按月、按年展开的带日期记录，并注入配置的伪影和干扰。**检查：** 病例通过发布检查才会放行。**运行：** 被测 agent 看到截至就诊时点 `T` 的记录，做出一个决定。**评分：** 代码把这个决定与由同一隐藏状态推出的金标比对，另有一个 LLM 裁判评判处理方案。

<p align="center">
  <img src="https://raw.githubusercontent.com/thetahealth/mirobody-env/main/docs/figures/readme_hero.png" alt="HAEnv 五步：定义隐藏的病人状态，渲染多年的记录，发布检查，被测 agent 回答题目，代码与 LLM 裁判对照由同一状态推出的隐藏金标判分" width="100%">
</p>
<p align="center"><sub>agent 只接收截至 <code>T</code> 的记录，隐藏金标只交给判分器；判分器是代码，另有一个 LLM 裁判判定处理方案质量 <code>Q</code>（见<a href="#计分">计分</a>）。</sub></p>

发布检查分三层：前提校验拒绝自相矛盾的 job 文件，逐项校验检查每道题的每条数据流和事件，泄漏探针检查每条题面。之后批次级闸门和题包审计检查整个题包，例如真实症状不能靠数据足迹与干扰事件区分开。仿真内核位于 `src/haenv_kernel/`。

- **题包是插件。** 每个题包把自己的金标、问法模板和计分登记为一个判据组；核心流水线不知道有哪些题包。
- **指纹跟着定义走。** 判分代码与出题代码各有一个指纹，打在每一行、每个批次上；指纹不一致的榜会被发布闸门拒绝。
- **四种形式。** 同一批病人可以用 `single`、`gated`（在预算内开检查）、`slices`（在多个时点分别提问）或 `multi`（分轮推进，agent 可修正）出题。

生成器在 job 声明的区间内为每个病人抽取药物反应、依从性与化验走势；难度（伪影、干扰、临床转折的时点）是 job 的设置。
[在线演示](https://thetahealth.github.io/mirobody-env/)在浏览器里生成病人，拖动参数即可改变病程、化验与依从性。

## 测什么

HAEnv（Health Agent Environment）是本仓库中的 Python 包与命令行工具 `haenv`。每道题要求在一份按月、按年增长的合成病历上做出一个临床决定。要用已发表的健康基准（如 ESL-Bench）评测现成系统，请用 [mirobody-eval](https://github.com/thetahealth/mirobody-eval)。

### 四个题包

| 题包 | `T` 处的决定 | 答案 |
|---|---|---|
| ① 限额诊断 | 哪些病能解释这份病历，包括藏在被第一种病解释掉的表现背后的第二种病；是否需要临床复核 | 鉴别诊断、在预算内要开的检查、复核标记 |
| ② 分诊 | 病人现在该去哪 | `ed_now` · `within_24h` · `routine_followup` · `watchful_waiting` |
| ③ 用药调整 | 这次随访时药该怎么办 | `uptitrate` · `downtitrate` · `maintain` · `switch` · `check_adherence_or_adverse_effect` |
| ④ 化验变化归因 | 与上次结果相比的变化是不是真的 | `true_change` · `analytic_biological_noise` · `preanalytical` · `method_difference`，以及真变化的幅度 |

每个题包的决定都由代码对照世界真值判定；题包 ① 按登记的别名把列出的诊断与金标比对，不调用模型。

- **先定金标。** 疾病、药物反应、依从性、化验真值与测量噪声在渲染病程之前确定，各题包的金标由代码从它们推出。
- **病人内部自洽。** 化验、用药、家庭与可穿戴数据流和生活事件由同一套规则在同一条时间线上生成；已知共病进入世界的表，在该改变答案的地方改变正确答案。
- **难度是参数。** 测量伪影、干扰事件、临床转折的时点都是 job 文件里的设置。
- **安全失误不会被平均掉。** 硬性门槛（越权改药、编造证据、漏掉急症、过早下结论）命中任何一道，该题记零分，其他得分无法抵消。

题包是管线，不是一份固定文件：一条命令按种子生成任意题量的题包并完成审计，全程不调用模型（见[快速开始](#快速开始)与[数据卡](https://github.com/thetahealth/mirobody-env/blob/main/docs/DATA_CARD.md#task-packs-and-their-generators)）。题面与病历中的自由文本为中文；字段名、答案枚举值与各类编号为英文。

## 排行榜

两个榜都在四个题包上排名，每包 100 题（N100），每题作答 3 次；当前榜是 1.3.1（十三个模型），其后是 1.3.0 榜（八个模型）。主分为每题三次作答的 `S = G·D·(1+Q)/2` 取平均（k3），四包等权；首次作答的预注册分 pk1 作为稳健性检查报告。95% 区间来自配对题簇自助法（2,000 次），对所有模型重抽同一批题目；名次区间来自同一批抽样。

### 1.3.1 榜单

1.3.1 榜单在同样的四个 N100 包上、用 1.3.1 的引用检查给 13 个模型排名，k3 聚合、区间和抽样次数都不变。
下面八个模型的诊断分与 1.3.0 表不同，因为在当前环境中刷新了受影响的作答，并应用当前评分器（见 `CHANGELOG.md`）。
分数读数：[`docs/boards/1.3.1/k3.json`](docs/boards/1.3.1/k3.json)，首次作答读数 [`docs/boards/1.3.1/pk1.json`](docs/boards/1.3.1/pk1.json)。

<!-- board-1.3.1:start -->
![13 模型总分、95% 区间与名次区间](docs/figures/readme_board_131.png)

| 名次 | 模型 | 总分（95% CI） | ① | ② | ③ | ④ | All-3 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1–2 | deepseek-v4.1-flash | 0.837 (0.813–0.861) | 0.655 | 0.932 | 0.928 | 0.833 | 0.823 |
| 1–2 | gpt-6-sol | 0.832 (0.808–0.856) | 0.693 | 0.893 | 0.959 | 0.785 | 0.878 |
| 3–5 | mimo-v2.6-flash | 0.793 (0.765–0.819) | 0.569 | 0.943 | 0.853 | 0.807 | 0.760 |
| 3–5 | gemini-3.8-flash | 0.792 (0.768–0.816) | 0.591 | 0.950 | 0.882 | 0.744 | 0.867 |
| 3–5 | qwen38-27b | 0.789 (0.763–0.815) | 0.617 | 0.930 | 0.885 | 0.722 | 0.765 |
| 6–7 | gpt-6-luna | 0.719 (0.692–0.747) | 0.494 | 0.859 | 0.897 | 0.627 | 0.693 |
| 6–7 | deepseek-v4-pro | 0.696 (0.666–0.724) | 0.533 | 0.933 | 0.689 | 0.629 | 0.668 |
| 8–10 | minimax-m3 | 0.633 (0.602–0.663) | 0.440 | 0.870 | 0.608 | 0.616 | 0.525 |
| 8–10 | deepseek-v4-flash | 0.632 (0.599–0.664) | 0.477 | 0.856 | 0.656 | 0.538 | 0.588 |
| 8–10 | glm-5.3-flash | 0.628 (0.597–0.658) | 0.484 | 0.813 | 0.668 | 0.547 | 0.530 |
| 11 | gemma-4-31b-it | 0.536 (0.506–0.567) | 0.382 | 0.767 | 0.514 | 0.482 | 0.580 |
| 12 | qwen3.7-flash | 0.446 (0.414–0.476) | 0.380 | 0.599 | 0.271 | 0.533 | 0.327 |
| 13 | qwen3-32b | 0.171 (0.150–0.192) | 0.069 | 0.307 | 0.058 | 0.249 | 0.073 |
<!-- board-1.3.1:end -->

### 1.3.0 总分

<p align="center">
  <img src="docs/figures/readme_board.png" alt="八个模型的总分与 95% 区间，按总分排列，左侧为名次区间" width="88%">
</p>

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

<sub>① 限额诊断 · ② 分诊 · ③ 用药调整 · ④ 化验变化归因。All-3 是三次作答的决定都正确的题目占比（四包等权平均），是可靠性画像，不是排名指标。`G` 表示通过硬门，`D` 表示核心决定正确，`Q` 是处理方案质量（同一 LLM 裁判两次判定的平均；区间不含裁判噪声）。名次区间重叠的模型不能确定先后。</sub>

评测所用题集以冻结题包文件发布，在各批次记录的提交上可逐字节重新生成。[交互式榜单](https://thetahealth.github.io/mirobody-env/#act3) ·
[榜单数据（JSON）](https://github.com/thetahealth/mirobody-env/blob/main/web/demo/board.json) ·
[早期榜单](https://github.com/thetahealth/mirobody-env/blob/main/docs/DATA_CARD.md#historical-boards) ·
[重算榜单](https://github.com/thetahealth/mirobody-env/blob/main/docs/REPRODUCE.md#recomputing-the-130-overall-board)

<!-- findings-1.3.1:start -->

## 1.3.1 发现

![错误诊断的证据覆盖与重新提问后的改对率](docs/figures/readme_findings_131.png)

**错误诊断大多缺证据。** 在列有决定性检查结果的题目上，错误诊断（不含中止）中 766 条未拿全决定性结果，247 条拿全了：75.6% 缺少部分决定性证据。

**补上漏掉的结果，多数能改对。** 从覆盖不全的错误诊断中抽取 142 条（gpt-6-luna 44 条、glm-5.3-flash 50 条、qwen3-32b 48 条），从最终作答之前重新提问。补上漏掉的决定性结果后 64.1% 改对；补上同样数量的非决定性结果为 12.7%，什么都不补为 8.5%（比非决定性结果高 51.4 个百分点，95% 区间 40.1 至 61.4）。在漏掉至少两项决定性结果的 79 条上，随机补一半改对 41.8%（不补 6.3%，全补 75.9%）。

**加大检查预算，各模型获益不同。** 100 道诊断题（首次作答）在一半、默认（即榜单）和两倍检查预算下的诊断正确率。增益：从一半到两倍预算的百分点变化及其配对 95% 区间；预算使用：平均花掉默认预算的比例。

| 模型 | 0.5× | 1× | 2× | 增益 [95% 区间] | 1× 预算使用 |
|---|---:|---:|---:|---:|---:|
| deepseek-v4-flash | 66% | 70% | 76% | +10.0 [2.0, 18.0] | 66% |
| gpt-6-sol | 71% | 77% | 81% | +10.0 [2.0, 18.0] | 73% |
| minimax-m3 | 55% | 59% | 65% | +10.0 [0.0, 19.0] | 44% |
| glm-5.3-flash | 55% | 63% | 63% | +8.0 [−2.0, 18.0] | 35% |
| deepseek-v4-pro | 71% | 77% | 77% | +6.0 [−1.0, 13.0] | 55% |
| gemini-3.8-flash | 72% | 77% | 76% | +4.0 [−4.0, 12.0] | 53% |
| gpt-6-luna | 63% | 67% | 66% | +3.0 [−3.0, 9.0] | 50% |
| qwen3.7-flash | 56% | 57% | 56% | 0.0 [−9.0, 8.0] | 36% |

**HTTP 重放复现已存工具结果。** 把榜单运行中 400 道题上 15,600 次作答会话的 256,609 次工具调用经 `haenv serve fhir` 重放并解码，256,609 次中 256,609 次与当前函数调用输出一致，并复现这些会话轨迹中存储的全部工具结果（含购买与拒绝）；返回的 4,621,363 个资源都是合法的 FHIR R4B（不合法 0 个）。重放工具是 [`tools/release/replay_http.py`](tools/release/replay_http.py)，在榜单池上运行：

```bash
python tools/release/replay_http.py --work leaderboard-pool/ --workers 32 --out artifacts/replay-http-results.json
```

预期汇总：`questions=400`、`responses=15600`、`calls=256609`、`identical=256609`、`differing=0`、`stored_differing=0`、`pending_purchases_replayed=234`、`validation_failures=0`；`missing=[]`、`worker_errors=[]`。榜单池含已存运行，需另行提供。源码树内的离线 HTTP 演练为 `python tools/release/replay_http.py --offline`。

离线重算并重评分公开的 re-ask 答案：`python tools/evaluation/reask_pooled.py --check`（[数据与条件](benchmarks/results/1.3.1/reask/README.md)）。
<!-- findings-1.3.1:end -->

## 快速开始

离线完成生成、校验、作答和计分，无需 API key，零成本。

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

每条命令退出码均为 0，`verify` 输出：

```
[haenv] generation: 3/4 case(s) passed the emission gate
[haenv] per-item verification: 82 item(s) · failed 0 · text leaked in 0 case(s)
```

第四例 `EWX-04` 以 `conflicts=['event_density_mismatch']` 被拒：它的高干扰档注入了 5 条症状事件，而它的症状率只允许 1 条。内置作业写入 `artifacts/`；`--fresh` 会开一个新批次，因为对已有作答的批次重出题，会让新题目配上旧作答计分。

<details>
<summary>生成一个任务题包</summary>

```bash
uv run --with scikit-learn --with joblib python tools/generation/make_pack.py --pack p4 --n 50 --out packs/p4
```

`--pack` 取 `m2`（①）、`p2`（②）、`p3`（③）或 `p4`（④）；`--n` 定题量，`--seed`（或 `HAENV_PACK_SEED`）定种子。同一题包、题量与种子出的包相同；批次只记种子的哈希。出包与审计各门全部通过时退出码为 0，审计没过的题包，`haenv run` 拒绝评测。示例 job、各类占比与审计见[数据卡](https://github.com/thetahealth/mirobody-env/blob/main/docs/DATA_CARD.md#task-packs-and-their-generators)。
</details>

<details>
<summary>用 FHIR REST 或 MCP 提供一道题</summary>

```bash
B=$(ls -d artifacts/results/early_warning/ew-demo/*/ | tail -1)   # 上面生成的批次
uv run haenv serve fhir --pack "$B" --question EWX-01 --port 8000 &   # 一道题，标准库 HTTP 服务
curl -s localhost:8000/metadata                                     # CapabilityStatement：路由与检索参数
uv run --extra serve haenv serve mcp --pack "$B" --question EWX-01  # 同一组工具，走 MCP stdio
```

服务端的检索、分页、行数上限、日期过滤和检查预算与函数调用工具一致，病例日换算成 ISO 日期；
`tools/release/replay_http.py` 通过它重放已存的工具调用，并核对解码后的输出完全一致。
</details>

<details>
<summary>从 PyPI 安装</summary>

```bash
pip install haenv
JOBS=$(python -c "from haenv import data_root;print(data_root()/'benchmarks'/'inputs')")
cd ~/my-workdir                     # 默认产物目录：./artifacts/
haenv build "$JOBS/example-ew.job.yaml" --gen deterministic --fresh
haenv run   "$JOBS/example-ew.job.yaml" --offline
```

`HAENV_OUTPUT_ROOT` 指定其他可写根目录；`HAENV_CONFIG_OVERLAY` 指向你自己的一个 YAML 文件，合并覆盖包内自带的 `config.yaml`（见[评测你自己的 agent](#评测你自己的-agent)）。
</details>

<details>
<summary>用真实模型评测（计费）</summary>

计费运行要一个上限和记录花费的共享账本，两个都得给，否则在发第一个请求之前就会停。密钥从 `HAENV_ENV_FILE` 指向的文件读取。

```bash
uv run haenv run benchmarks/inputs/example-ew.job.yaml --models gemini-3.1-pro --limit 1 \
  --judge-budget-usd 5 --judge-budget-ledger ~/.haenv/budget.json
```

评测默认断点续跑，只补发还没有答案的格子。缓存、token 预算与用量记账见 [`docs/REPRODUCE.md`](https://github.com/thetahealth/mirobody-env/blob/main/docs/REPRODUCE.md#3-the-step-that-costs-money)。
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

agent 返回风险预测、从固定列表中选出并排序的驱动因素（各附证据编号），以及 `A0`（继续观察）到 `A5`（紧急升级）之间的一个动作类别。本例的隐藏金标：体重会回升，驱动因素是 `poor_medication_adherence`，且需要临床介入。

| 离线参照求解器 | 预测 | 首位驱动因素 | 动作 | 结果 |
|---|---|---|---|---|
| `no_revision` | 风险 0.2，低 | `poor_medication_adherence`（命中） | `A0`，未请求复核 | 命中硬性门槛 `premature_closure`，整例记零分 |
| `const_ddx` | 风险 0.5，无法判断（弃权） | `unknown_or_multifactorial` | `A3`，请求复核 | 正常计分 |

第一个求解器找对了驱动因素，整例仍然判负：本例需要临床介入，它却选择继续观察。
</details>

## 计分

- **作答分。** 每次作答记 `S = G·D·(1+Q)/2`。四个题包的硬门 `G` 与核心决定 `D` 都由代码判定；一个 LLM 裁判（`gpt-6-luna`）只判处理方案质量 `Q`。题包分是各题 `S` 的平均。
- **重算而非重跑。** 模型原始回答全量保存，修正判分只需重算（`tools/evaluation/restamp_batch.py <batch>`），无需再次调用被测模型。
- **可单独审阅的核心。** `src/verifier_core/` 包含指纹、发布闸门、硬性门槛乘子和噪声下限审计，不含临床词汇。

门槛清单、未发布的 2026-10-06 候选版的题包综合分，以及计分、画像、退役三类度量见[数据卡](https://github.com/thetahealth/mirobody-env/blob/main/docs/DATA_CARD.md#scoring)。自检不调用模型，几秒内完成，见 [`docs/REPRODUCE.md`](https://github.com/thetahealth/mirobody-env/blob/main/docs/REPRODUCE.md#4-self-checks)。

## 评测你自己的 agent

被测模型是 `config.yaml` 里 `models:` 下的一项，可以接任何在 `backends:` 中声明的 OpenAI 兼容接口。它拿到同样的病人、同样在 `T` 处截断、同样的判分器；挂在这类接口后面的 agent 也用同样方式评测。新接口写在 `config.yaml` 旁边的 `config.local.yaml` 里（`pip install` 安装时写进 `HAENV_CONFIG_OVERLAY` 指向的文件）；密钥从 `HAENV_ENV_FILE` 指向的文件读取：

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

```bash
uv run haenv run benchmarks/inputs/example-ew.job.yaml --models my-agent --limit 1 \
  --judge-budget-usd 5 --judge-budget-ledger ~/.haenv/budget.json
```

带 `--limit` 时是对接口的抽样检查，运行以退出码 6 结束，表示得分不完整。完整运行还需要 OpenRouter 密钥供 `Q` 裁判使用，裁判费用计入同一个预算。模型路由与价格见 [`docs/REPRODUCE.md`](https://github.com/thetahealth/mirobody-env/blob/main/docs/REPRODUCE.md#model-routes-and-prices)。

| 要改的 | 位置 | 是否写代码 |
|---|---|---|
| 被测模型或 agent | `config.yaml`（`models:`、`backends:`） | 否 |
| 新增判分器或新的金标类型 | 登记一个函数 | 是 |
| 判分器观察的内容 | 被观察对象插件 | 是 |
| 数据流、事件、药效、伪影 | 环境侧插件 | 是 |

[`examples/`](https://github.com/thetahealth/mirobody-env/blob/main/examples/README.md) 中有五个插件示例包，各配一个负对照。指南：[判分器插件](https://github.com/thetahealth/mirobody-env/blob/main/docs/design/llm-judge-plugin.md) · [外部任务类型](https://github.com/thetahealth/mirobody-env/blob/main/docs/design/external-task-contract.md)。

## 文档

| 主题 | 位置 |
|---|---|
| 技术规格、评分细节与补充结果 | [`docs/supplement.md`](https://github.com/thetahealth/mirobody-env/blob/main/docs/supplement.md) |
| 题包、金标来源、计分细节、早期榜单、已知缺口 | [`docs/DATA_CARD.md`](https://github.com/thetahealth/mirobody-env/blob/main/docs/DATA_CARD.md) |
| 孪生病历：只改一处声明字段的成对题目 | [`docs/DATA_CARD.md#twin-records`](https://github.com/thetahealth/mirobody-env/blob/main/docs/DATA_CARD.md#twin-records) |
| 哪些可免费复现、哪些计费、缓存与自检 | [`docs/REPRODUCE.md`](https://github.com/thetahealth/mirobody-env/blob/main/docs/REPRODUCE.md) |
| 从对话里驱动流水线（三份 agent 技能） | [`docs/README.md#agent-skills`](https://github.com/thetahealth/mirobody-env/blob/main/docs/README.md#agent-skills) |
| 仓库结构与设计文档 | [`docs/README.md`](https://github.com/thetahealth/mirobody-env/blob/main/docs/README.md) |
| 数据来源与使用边界 | [`docs/ETHICS.md`](https://github.com/thetahealth/mirobody-env/blob/main/docs/ETHICS.md) |
| 如何修改计分或生成代码 | [`CONTRIBUTING.md`](https://github.com/thetahealth/mirobody-env/blob/main/CONTRIBUTING.md) |

## 局限

- **包 ④ 的 RCV 基线。** 只读两次印出值和规程 RCV 表的读者，在包 ④ 上 D 约 0.5，所以 ④ 分数在 0.5 附近只说明会用 RCV。
- **名次不确定性。** 1.3.1 榜的 12 对相邻模型中，只有 6 对的配对分差区间不含 0；区间重叠时不能确定先后。
- **临床复核。** 题包 ② 和 ③ 背后的规则以及若干登记条目带 `review: pending`；医学内容整体尚未经执业医生审阅。
- **单一语义裁判。** 一个模型（`gpt-6-luna`）为 `Q` 评分，被测模型中有同一厂商的模型。跨厂商的 Gemini 裁判（服务方实际提供 `gemini-3.8-flash`，它本身也是被测模型）在历史 1.3.0 榜上重判了全部 6,937 条 `G·D = 1` 的作答：八个模型的名次不变，模型层面平均 `Q` 与主裁判的相关 r = 0.968。`Q` 的量表在本榜作答上校准，没有临床医生核验。
- **合成数据。** 所有病人都是合成的，不构成任何医疗建议。

完整清单见[数据卡](https://github.com/thetahealth/mirobody-env/blob/main/docs/DATA_CARD.md#known-gaps)。

## 相关工作

| 工作 | 评测内容 |
|---|---|
| ESL-Bench（[arXiv:2604.02834](https://arxiv.org/abs/2604.02834)，[数据集](https://huggingface.co/datasets/mirobody/ESL-Bench)） | 100 名合成用户，各有 1–5 年的设备、体检与事件轨迹，答案由程序计算；排行榜：[Health Memory Arena](https://healthmemoryarena.ai) |
| mirobody-eval（[GitHub](https://github.com/thetahealth/mirobody-eval)） | 用已发表的健康基准评测现成系统的框架 |
| MedAgentBench（[arXiv:2501.14654](https://arxiv.org/abs/2501.14654)） | FHIR 虚拟 EHR 中的 300 个 agent 任务 |
| EHRSHOT（[arXiv:2307.02028](https://arxiv.org/abs/2307.02028)） | 6,739 名真实病人纵向 EHR 上的少样本预测 |
| LongHealth（[arXiv:2401.14490](https://arxiv.org/abs/2401.14490)） | 20 份长篇虚构病历上的 400 道选择题 |
| HealthBench（[arXiv:2505.08775](https://arxiv.org/abs/2505.08775)） | 5,000 段健康对话，由模型按医生编写的细则评分 |
| AgentClinic（[arXiv:2405.07960](https://arxiv.org/abs/2405.07960)）、CRAFT-MD（[doi:10.1038/s41591-024-03328-5](https://doi.org/10.1038/s41591-024-03328-5)） | 与 LLM 扮演的病人对话，完成问诊与诊断 |

HAEnv 同时具备在 `T` 处截断的纵向病历、先于数据确定的金标、决定与硬门由代码计分、处理方案质量由一个 LLM 裁判判定，以及可重新生成的合成病例。

## 数据、伦理与复现

> 整条管线公开：渲染病人的世界、写题的生成器、判分器。canary 串（[`CANARY.md`](https://github.com/thetahealth/mirobody-env/blob/main/CANARY.md)）标记本题包的文本，便于检测它是否出现在某个语料中；允许用公开样例训练；1.3.0 榜的题由公开种子生成（更早的榜用私有种子）。

所有病人均为合成数据，不构成医疗建议，不可用于临床决策。

## 引用

```bibtex
@software{haenv,
  title  = {Health Agent Environment},
  author = {{Theta Health}},
  year   = {2026},
  url    = {https://github.com/thetahealth/mirobody-env},
  version = {1.3.1}
}
```

同样的元数据见 [`CITATION.cff`](https://github.com/thetahealth/mirobody-env/blob/main/CITATION.cff)。

## 许可与致谢

代码采用 [MIT 许可](https://github.com/thetahealth/mirobody-env/blob/main/LICENSE)；合成数据与题包采用 [CC BY 4.0](https://github.com/thetahealth/mirobody-env/blob/main/LICENSE-DATA)。第三方声明见 [`NOTICE.md`](https://github.com/thetahealth/mirobody-env/blob/main/NOTICE.md)。

HAEnv 的灵感来自 [ESL-Bench](https://arxiv.org/abs/2604.02834)。

<p align="center"><sub>合成数据 · 仅供评测 · 非医疗建议</sub></p>

<details>
<summary>让一次答错的诊断回答再答一次</summary>

`haenv reask` 从存档对话最终答案之前的位置继续，选择有证据清单、已作答且诊断错误的回答，
再向原模型各问一次：`resample` 不补结果；`decisive` 补携带遗漏证据的结果；`placebo` 补同等
数量、且不携带任何证据清单条目的结果；`half` 按种子抽取 `floor(k/2)` 条 decisive 结果，
只用于 `k >= 2` 的单元，并保留原顺序。证据已齐全但仍答错的回答只运行 resample。

```bash
haenv reask saved-batch/ --models glm-5.3-flash --questions QUESTION_ID \
  --sources saved-result-corpus/ --dry-run
haenv reask saved-batch/ --models glm-5.3-flash --questions QUESTION_ID \
  --sources saved-result-corpus/ --out reasked/ --model-cap glm-5.3-flash=32 \
  --judge-budget-ledger reask-ledger --judge-budget-usd 1
```

源目录需要 `query_trace.jsonl`、`payloads.jsonl`，以及 `responses.jsonl` 或计费回执中的最终请求
指纹；`trace.jsonl` 的购买观察值用于重建工具声明。未存档的 decisive 购买通过环境自身购买流程
生成，需要 `cases.jsonl`。`--sources` 补充同题的已存档返回；要复现选择，须使用相同来源集合、
`--seed` 和 `--half-seed`。原请求指纹不匹配或 placebo 不足等量时，在调用模型前明确失败。
placebo 排除所有尝试买过的项目（包括被拒绝的购买），也可选不携带证据项的监测流。

各注入组统一使用空文本 assistant 工具调用消息和原样 tool 返回，不向模型暴露证据标签。
最后关闭工具；GLM Flash 路由为兼容使用 `auto`，若返回工具调用则记为未作答且不执行。
`--dry-run` 输出每组测试名称和携带的证据项，不初始化模型提供方。

结果目录包含 `injection-rows.jsonl`（`model/question/repeat/kind/arm/status/D/n_added`，
组代码 `A/B/C/H`）、完整请求及选择清单、答案、调度回执与正常计费记录。
`D` 由随包诊断判分器对存档 gold 计算；传输失败为 `D: null`。已完成组可续跑，计划变化须使用新
输出目录。repeat 优先取存档字段，再取目录 `-pkN`/`-repN` 后缀，否则为第一次。

默认每模型 64 并发，将历史较小的 cell cap 提升至该并发数，同时受共享付费请求容量约束；
显式 `--model-cap MODEL=N` 优先。Python API 为 `haenv.reask.prepare(...)`、`dry_rows(units)`
和 `run(units, output, ledger_path=..., budget_usd=..., model_caps=...)`，规则与 CLI 相同。

</details>
