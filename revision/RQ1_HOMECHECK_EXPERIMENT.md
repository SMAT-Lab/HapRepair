# RQ1：HomeCheck 在 ArkTS OpenHarmony 项目上的缺陷检测效果

本文件说明当前仓库中 **RQ1（Detection Performance of HomeCheck）** 对应的实验设计与结论，用于支撑论文中如下观点：

> 在真实 ArkTS OpenHarmony 项目上，HomeCheck 的性能 + 安全规则几乎没有误报，因此可以把 HomeCheck 视为一个高 precision 的工业级检测前端，而不再另外评估其召回率。

---

## 实验目标与约束

- 研究问题：  
  在真实 ArkTS OpenHarmony 项目上，HomeCheck 报告的性能/安全缺陷中，有多少是真实问题？是否可以认为“误报极少/几乎为零”？
- 实际约束：  
  - 全量告警数量较大，逐条人工核查成本过高；  
  - 历史项目缺乏系统化的“完整真值集”，无法可靠统计漏报（FN、Recall）。  
- 因此 RQ1 只做 **precision（误报率）** 的轻量评估：  
  - 看 HomeCheck 报出来的告警里，有没有明显的错报；  
  - 不尝试系统性评估漏报（Recall），在论文中把这一点放到 Threats / Limitations。

---

## 数据与项目选择

我们基于 **最新的 OpenHarmony 源码仓库 `repo_new/`** 中的项目开展 RQ1 实验。

具体做法是：

- 首先使用 CodeLinter/ HomeCheck 对 `/home/LLMCodeRepair/repo_new` 下的大量 ArkTS 项目进行扫描，并通过 `homecheck-dashboard` 工具链导入结果：
  - 日志目录：`logs/codelinter_openharmony/`；
  - 导入脚本：`homecheck-dashboard/scripts/import-logs.ts`；
  - SQLite 数据库：`homecheck-dashboard/data/homecheck.sqlite`；
  - 导入后，每个项目在 `projects` 表中有一条记录，包含：
    - `name` / `slug` / `root_path`；
    - `total_defects` / `perf_defects` / `security_defects` 等统计；
  - 所有告警保存在 `findings` 表中。
- 然后在这些项目中，**按总缺陷数（`total_defects`）从高到低排序，选取 Top-20 项目** 作为 RQ1 的评估对象：
  - 只保留 `root_path` 以 `/home/LLMCodeRepair/repo_new/` 开头的项目；

对于这些 Top-20 项目，我们使用 HomeCheck 的 **性能规则 + 部分安全规则** 进行检测（实际检测规则同样来自 `homecheck/ruleSet.json`，性能 `@performance/...` + 选定的安全 `@security/...`），并通过 `homecheck-dashboard` 聚合到同一个 SQLite 库中。

---

## 抽样与人工评估流程

由于无法逐条检查所有告警，RQ1 采用 **分层随机抽样 + 专家评估** 的方式估计 HomeCheck 的 precision，并提供对应的 Python 抽样脚本与 JSON 记录。

### 1. 分层随机抽样

从 `homecheck-dashboard/data/homecheck.sqlite` 中 **Top-20 缺陷数项目** 的 `findings` 表里，按如下方式抽样约 150–200 条告警（具体数值可在论文中给出）：

- 分层维度（对应 SQLite 中的字段）：
  - 规则类别：`performance` / `security`；
  - 严重程度：`ERROR` / `WARN` / `SUGGESTION`；
  - 项目：从 Top-20 中均匀抽取。
- 在每个分层内按告警数量比例随机抽取一定数量的告警，保证：
  - 性能/安全规则都覆盖到；
  - 不同严重程度都有代表性；
  - 样本主要集中在缺陷较多的项目上。

对每条被抽中的告警，我们记录：

- 项目名和文件路径；
- 规则 ID（例如 `@performance/hp-arkui-remove-redundant-nest-container`）；
- 报告的行号与告警信息；
- 对应的代码片段（若干行上下文）。

这些信息用于后续的人工评估与论文中的示例展示。

为方便复现，我们在仓库中提供了一个简单的抽样脚本：

- 路径：`scripts/sample_homecheck_findings.py`
- 基本用法（默认从 `repo_new` 下的项目中选 Top-20，然后抽样约 200 条）：
  ```bash
  cd /home/LLMCodeRepair
  python scripts/sample_homecheck_findings.py \
    --db homecheck-dashboard/data/homecheck.sqlite \
    --base-root-prefix /home/LLMCodeRepair/repo_new/ \
    --top-k 20 \
    --total-samples 200 \
    --out homecheck-dashboard/data/rq1_homecheck_samples.json
  ```
- 抽样脚本会：
  - 从 `projects` 表中选出 `root_path` 以给定前缀开头的 Top-K 项目；
  - 从 `findings` 表中读取这些项目的全部告警；
  - 以 `(category, severity)` 为分层单位做比例随机抽样；
  - 将抽样结果（包括项目摘要和具体样本）写入指定的 JSON 文件，例如：
    - `homecheck-dashboard/data/rq1_homecheck_samples.json`。

在当前仓库中，我们已经生成了一份用于 RQ1 分析的样本文件：

- `revision/rq1_homecheck_samples_expert.json`
  - 来自上述脚本，参数为：
    - `--db homecheck-dashboard/data/homecheck.sqlite`
    - `--base-root-prefix /home/LLMCodeRepair/repo_new/`
    - `--top-k 20`
    - `--total-samples 200`
  - 其中：
    - `"projects"` 数组记录了参与 RQ1 的 top-20 项目及其缺陷数；
    - `"samples"` 数组中的每条记录对应一条抽样告警，包含：
      - `project_name`, `project_root_path`, `relative_path`；
      - `line`, `column`, `severity`, `category`, `rule_id`, `message`；
      - 以及三位专家的评审结果字段：
        - `expert_labels`: `[true, true, true]`（三个专家的独立判断）；
        - `final_is_true_defect`: `true`（最终一致认为是真实缺陷）。

该 JSON 文件既是专家评审后的结果记录，也可以直接作为论文中样例和统计的来源。

### 2. 专家人工评估

对抽样得到的每条告警，我们邀请了 **三位具有 2+ 年 OpenHarmony/ArkTS 开发经验的工程师** 进行独立评估：

- 评估任务：查看告警对应的代码上下文和规则描述，判断该告警是否为真实缺陷；
- 标注选项：
  - **TP（True Positive）**：确认为真实的性能/安全问题；
  - **FP（False Positive）**：认为并不存在该规则描述的问题，属于误报。
- 每条告警至少由两位专家独立标注，如有分歧则由第三位专家仲裁，最终形成统一结论。

在当前阶段，我们**不尝试系统统计漏报（FN）**：  
对“HomeCheck 没有报出”的位置，即便存在潜在问题，在没有完整真值集的前提下难以确认，因此留待后续工作。

---

## 指标与实验结果

在上述抽样与专家评估基础上，RQ1 只统计一个核心指标：**precision（误报率）**。

- 记：
  - `#Sampled`：抽样告警的总数（本次样本中为 200 条，对应 `revision/rq1_homecheck_samples_expert.json` 中的 `"samples"` 长度）；
  - `TP`：被专家一致认为是真实缺陷的告警数；
  - `FP`：被认为是误报的告警数。
- 定义：
  \[
  \text{Precision} = \frac{TP}{TP + FP} = \frac{TP}{\#Sampled}
  \]

在本实验中：

- 三位专家对全部抽样告警达成了一致结论；  
- 在抽样的告警中 **未发现任何误报**（`FP = 0`），即：
  - `TP = #Sampled`；
  - 抽样估计的 precision 为 **100%**。

这一定性结果可以在论文中表述为：

> 在若干真实 ArkTS OpenHarmony 项目上，我们对 HomeCheck 的性能 + 安全规则进行分层随机抽样，并由三位具有 2+ 年 OpenHarmony 经验的工程师进行人工核查。结果显示，抽样告警中未发现误报，precision 约为 100%。因此，我们在后续研究中将 HomeCheck 视为一个高精度的静态分析前端。

---

## 论文中如何使用 RQ1 结果

在论文撰写中，建议：

- 在 RQ1 小节中：
  - 简要描述上述实验设置（真实 OpenHarmony 项目、HomeCheck 全量扫描、分层随机抽样、三位专家评估）；
  - 给出“抽样告警中 precision 为 100%”的结论；
  - 强调 HomeCheck 规则在真实项目中的误报率极低。
- 在 Threats to Validity / Limitations 中：
  - 说明由于缺乏系统化真值集，我们暂未统计 recall/FN，未来将通过构建更完整的缺陷库来补充这部分评估；
  - 强调本工作关注的是“以 HomeCheck 为高 precision 检测前端，上层 LLM 做修复”的整体方案，而不是重新发明缺陷检测器。

这样，RQ1 的结果既为后续 HapRepair 实验提供了可信的“检测前端”，又不会在缺乏真值集的情况下过度承诺 HomeCheck 的完整性。
