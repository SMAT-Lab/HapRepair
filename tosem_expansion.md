## TOSEM 扩展实验设计说明（HomeCheck + HapRepair）

本文档说明在现有 FSE 论文实验基础上，如何系统地扩充为 TOSEM 版本，对应 `results.tex` 中的 RQ1–RQ5。你后续只需要按本说明跑实验、填表和补充分析即可。

---

## 总体目标

- 把 **HomeCheck** 提升为一等公民的静态分析贡献：既有架构设计，也有**检测效果**实证。
- 用 **扩展规则集（performance + 部分 security）** 和 **更丰富的对比对象**，支撑以下 5 个 RQ：
  - RQ1：HomeCheck 在 ArkTS OpenHarmony 项目上的缺陷检测效果。
  - RQ2：HapRepair 在 HomeCheck 检出的缺陷上的整体修复效果。
  - RQ3：组件/设置（上下文、RAG、变化信息等）的消融。
  - RQ4：不同 LLM（GPT-4o / Llama-3.2-70B / Qwen-2.5-72B）的对比。

---

## 数据集与规则扩展


### 1. 规则选择

- **Performance 规则**  
  - 继续沿用现有 HomeCheck 中使用的性能规则集（与现在的 271 对示例一致）。

- **Security 规则（新增）**  
  - 从 HomeCheck 的 security 规则中选取 **10–20 条代表性规则**，建议覆盖：
    - 权限/隐私相关（如 GPS、摄像头、麦克风等）。
    - 网络/文件 I/O 资源使用。
    - 重要 API 滥用或错误配置。
  - 在论文中作为“代表性 security 规则”，不要尝试覆盖所有 security 规则。

### 2. 知识库扩展（用于 RQ3–RQ5 修复）

- 对每条选中的 security 规则，按现有 performance 规则的格式，构建若干 **缺陷–修复对**：
  - buggy 代码片段；
  - 对应的修复代码；
  - 自然语言解释；
  - diff 变化信息。
- 将这些样本加入现有的 271 对示例库中（最好在代码中标记规则 ID，便于统计）。
- 在 `results.tex` 中，我们已经用 `\tosem{}` 标明：知识库**主要**基于 performance 规则，同时加入若干代表性 security 规则。

### 3. 测试项目与缺陷标注

- 继续使用现有的 OpenHarmony / SIG / TPC 仓库集合，筛选出：
  - **至少 10 个 performance 或 security 缺陷**的项目。
- 对于加入 security 规则后，新检测出的缺陷：
  - 人工检查 HomeCheck 报告，确认哪些是真实缺陷；
  - 如发现 HomeCheck 漏报，人工记录这些漏报缺陷，用于后续 recall/FN 统计。
- 得到一个“扩展缺陷集”：
  - 覆盖 performance + 选定的 security 规则；
  - 每条缺陷记录：项目、文件、位置、规则 ID、是否真实缺陷。

---

## RQ1：HomeCheck 检测效果（Detection Performance of HomeCheck）

### 目标

- 在 **真实 ArkTS OpenHarmony 项目**上，给出 HomeCheck 在 performance + security 规则上的**误报情况（precision）**的轻量评估，支撑“规则质量较高、几乎零误报”这一前提假设，而不是做一个大规模的完整检测效果基准。

### 实验步骤

1. 使用 HomeCheck 扫描所有选定的 ArkTS OpenHarmony 项目，开启：
   - 所有性能规则；
   - 选定的 security 规则。
   得到“全量告警集合”（只用于规模感描述，不全部人工标注）。
2. 在这些项目中，选取若干**具有代表性、缺陷数较多**的项目（例如：总缺陷数排名前若干名的项目，或每个类别各选若干项目）。
3. 在选定项目的告警中，按如下方式进行**分层随机抽样**，得到约 150–200 条需要人工检查的告警：
   - 规则类别分层：performance / security；
   - 严重程度分层：error / warning / suggestion；
   - 在每个分层内随机抽取若干条（保证每类都有一定数量样本）。
4. 使用我们当前的 Web 标注界面，对抽样到的每条告警进行人工核查：
   - 标记为 **TP**（真实缺陷）或 **FP**（误报）；
   - 可选：记录简单说明（如“边界情况”“项目特定约定”）。
5. 作为补充，可以对个别规则或项目再做少量“便利检查”（non-random spot-check），验证抽样结论在未抽样告警上的一致性，但不纳入正式统计。

### 指标与表格建议

- 统计维度：
  - **按规则类别**（performance / security）；
  - **按项目**（可选，用于分析差异）。
- 指标：
  - 在当前阶段，我们**优先统计 Precision**：即 HomeCheck 报告的告警中有多少是真实缺陷（TP vs FP）。  
  - 只在抽样子集上估计 precision，不尝试系统性统计 FN / Recall。
- 推荐表格（结果出来后填进去）：
  - `Table~\ref{tab:homecheck_detection}`：  
    - 列：Rule Category / #AllReports / #Sampled / TP / FP / Precision（如未来补齐 FN，再扩展为 Recall / F1）。
    - 行：Performance / Security / Overall（可选：按严重程度再拆几行）。
  - 若篇幅允许，可在附录中给出按具体规则的细粒度统计。
- **说辞与合理性**：
  - HomeCheck 的规则来源于我们在真实 OpenHarmony 项目中的长期实践，总结了常见的性能/安全隐患，因此 RQ1 的核心问题是“这些经验规则在真实项目里的可用性如何？”
  - 由于历史项目缺乏成体系的缺陷真值集，且全量告警数量很大，逐条人工标注成本过高、也难以系统性统计漏报（FN），我们选择：
    - 通过分层随机抽样 + 人工核查，给出 HomeCheck 在真实项目上的 **precision 估计**，验证“误报率极低”的经验结论；
    - 不追求完整的 recall 评估，而是把“漏报分析”留在威胁与局限/未来工作中说明。
  - 在论文中要显式声明这一现实局限，并补充两类佐证：
    1. 抽样告警的人工核查结果，显示 HomeCheck 在 performance / security 规则上误报率都较低（例如 precision 接近 100% 或 > 98%）；
    2. 展示我们构建的 `data/MyApplication2/entry/src/main/ets/securitycases/` 正反例和若干真实项目案例，证明规则在受控环境和实战场景中都能检测到真实问题。
  - 在“威胁与局限”小节中说明：缺乏系统性 FN 数据是当前行业普遍困境，我们会在未来工作中逐步构建更完整的安全缺陷库以补充召回评估；本工作把 HomeCheck 视为一个 **高 precision 的工业级检测前端**，后续 RQ3–RQ5 的关注点在于缺陷修复而非重新发明检测规则。

### 论文中对应位置

- `results.tex` 中的：
  - `\subsection{\tosem{RQ1: Detection Performance of \static{}}}`；
  - `\answer{RQ1}{...}`：把“(To be completed...)”替换为一两句总结，例如：
    - HomeCheck 在 performance 规则上精度很高，security 规则略低但仍然可用；
    - 哪类规则更容易误报/漏报。

---

## RQ2：HomeCheck vs SOTA Coding Agent

### 目标

- 比较 HomeCheck 与一个 SOTA ArkTS Coding Agent 的检测能力，特别是 **多缺陷文件**场景。

### Coding Agent 设计

- 基于 GPT-4o 构建一个 agent：
  - 输入：单个 ArkTS 文件（可以限制长度，必要时按函数拆分）；
  - 额外输入：当前启用的 HomeCheck 规则的自然语言描述；
  - 输出：该文件中每个缺陷的位置（行/列）和规则类型（或缺陷类型描述）。
- Prompt 设计要尽量接近评审能接受的“coding agent”，例如：
  - 先告诉模型它是 ArkTS 性能/安全缺陷检测专家；
  - 然后给出数个示例（ICL），再给出目标文件，让其列出所有缺陷位置。

### 数据与匹配

- 使用与 RQ1 相同的扩展缺陷集（performance + security）。  
- 重点选择：
  - 部分**单缺陷文件**；
  - 部分**多缺陷文件**（包含多规则、多位置）。
- 对 agent 输出与 ground truth 做匹配：
  - 位置允许一定误差（例如同一行或邻近几行算匹配）；
  - 缺陷类型按规则 ID 或类型描述匹配。

### 指标与分析

- 同样统计 `TP, FP, FN, Precision, Recall, F1`：
  - 按工具：HomeCheck / Agent；
  - 按文件类型：Single-defect / Multi-defect。
- 推荐表格（结果出来后填进去）：
  - `Table~\ref{tab:homecheck_vs_agent}`：
    - 列：Setting（Single-defect / Multi-defect） / Tool / TP / FP / FN / Precision / Recall / F1。
    - 补充一行 Overall。

### 论文中对应位置

- `results.tex` 中的：
  - `\subsection{\tosem{RQ2: Comparison with a Coding Agent}}`；
  - `\answer{RQ2}{...}`：总结类似：
    - HomeCheck 在 recall、特别是多缺陷文件上的优势；
    - Coding Agent 容易漏掉后面的缺陷，或产生较多 FP。

---

## RQ3：Repairing Performance（扩展规则集上的修复）

### 目标

- 在**扩展规则集（performance + 部分 security）**上评估 HapRepair 的修复效果。

### 实验设置

- 使用扩展后的 HomeCheck 缺陷集作为输入：
  - 所有 performance 缺陷；
  - 选定 security 规则的缺陷。
- 修复流程与现有实验保持一致：
  - HomeCheck 定位缺陷；
  - 构造上下文 + 检索类似示例；
  - LLM（默认 GPT-4o）生成补丁；
  - 应用补丁 + CFG 验证；
  - 重新运行 HomeCheck，迭代多轮修复。

### 输出与分析

- 更新 `Table~\ref{tab:project_statistics}` 与 `Table~\ref{tab:rule_statistics}`：
  - 行可以拆分为 `performance` vs `security`，或者在每类中再细分；
  - 观察 security 缺陷的修复率是否显著低于 performance。
- 文本分析中说明：
  - 在更广泛的规则上，整体修复率是否依然接近 99%；
  - 哪些 security 规则难修复（需要人工干预或更多知识）。

### 论文中对应位置

- `\subsection{\tosem{RQ3: Repairing Performance}}`；
- `\answer{RQ3}{...}`：更新总结为“在 performance+security 扩展集上的整体修复能力”。

---

## RQ4：Ablation Study（扩展数据集上的消融）

### 目标

- 在 **扩展后的缺陷集** 上，分析：
  - 上下文策略（局部 vs 整文件）；
  - RAG 检索的 top-k；
  - 是否加入 diff/变化信息；
  的影响。

### 实验注意点

- 保持与当前表格一致（`Table~\ref{tab:retrieval_numbers}`、`Table~\ref{tab:context_strategy}`、`Table~\ref{tab:augmentation_methods}`），但：
  - 样本项目应来自扩展后的数据集；
  - 指标仍然是“剩余缺陷数 + 修复率”，但可以加一列区分 performance / security。

### 论文中对应位置

- `\subsection{\tosem{RQ4: Ablation Study}}` 中已经加了一句：
  - “默认在扩展数据集上进行消融”；
- `\answer{RQ4}{...}`：根据新结果，适当提到 security 规则上的表现差异。

---

## RQ5：Different LLMs in HapRepair

### 目标

- 比较 GPT-4o / Llama-3.2-70B / Qwen-2.5-72B 在扩展规则集上的修复能力差异。

### 实验设置

- 使用与 RQ3 相同的缺陷集和 HomeCheck 配置；
- 仅替换 LLM，并保持：
  - 同样的 prompt 模板；
  - 同样的 RAG 检索策略；
  - 同样的迭代轮数。

### 输出与分析

- 更新现有的 LLM 对比表（`Table~\ref{tab:project_statistics_llama3.2}` 和 `Table~\ref{tab:project_statistics_qwen2.5-72b-instruct}`）到扩展数据集；
- 在 `\answer{RQ5}{...}` 中总结：
  - 是否存在明显的“第一梯队”与“次优梯队”；
  - 对 security 规则修复的差异（如果有）。

---

## 实施 checklist

1. 选定代表性 security 规则，并在 HomeCheck 中开启这些规则。
2. 扫描测试项目，人工标注 TP/FP/FN，构建统一 ground truth。
3. 跑 RQ1 HomeCheck 检测实验，填 `Table~\ref{tab:homecheck_detection}`。
4. 构建 Coding Agent，跑 RQ2 对比实验，填 `Table~\ref{tab:homecheck_vs_agent}`。
5. 为 security 规则补充若干修复示例，加入知识库。
6. 用扩展规则集重跑修复实验（GPT-4o 为主），更新项目维度和规则维度的修复表。
7. 在扩展集上重跑 ablation 实验和不同 LLM 对比实验，更新相关表格。
8. 修改 `\answer{RQ1}`–`\answer{RQ5}` 中的内容，将占位文字替换为实验结论。

只要按以上步骤推进，等实验结果出来后，你只需要填数字和写结论，不再需要重新设计实验整体结构。  
