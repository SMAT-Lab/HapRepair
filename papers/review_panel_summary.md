# 大模型评审团意见汇总（面向 TOSEM 扩展稿）

本文档汇总以下三份评审文本中的主要共识与分歧，并整理为可执行的 TOSEM 改稿清单：

- `papers/paperreview.md`（英文审稿意见）
- `papers/GPTreviewer.md`（中文翻译/转述）
- `papers/GeminiReviewer.md`（中文审稿意见）

同时，结合当前仓库中的 TOSEM 扩展稿（`-FSE-Industry2025-Learn-to-Repair-OpenHarmony-Apps/main.pdf` 与对应 LaTeX），标注“已覆盖/仍欠缺”的关键点。

---

## 1. 总体共识（评审团一致认可的点）

### 1.1 选题与动机
- 低资源语言/平台（缺乏测试、训练语料、成熟工具链）下的 APR 是重要且现实的 SE 问题。
- 论文把研究重心从“更强的补丁生成”转到“在无测试条件下构造可靠 oracle（判定器）”是有价值的定位转变。

### 1.2 方法论主线
- “static-oracle–guided LLM repair”的范式清晰：将 *where-to-fix*（高精度静态分析定位 + 迭代验证）与 *how-to-fix*（LLM 在语义不确定下生成补丁）解耦。
- 以高精度静态分析作为迭代反馈源（frozen rule set/configuration 下重复 re-check）是 test-scarce 场景下可扩展的工程路径。

### 1.3 系统与实证（在论文定义的指标下）
- 架构组件（HomeCheck + 上下文抽取/合并 + RAG + diff scaffolding）对“知识稀缺”和“长上下文/多缺陷文件”的痛点有针对性。
- 大规模评测（35 projects、8,664 findings、5 iterations、多模型 + 消融）在“静态规则合规/remaining findings”指标下具有说服力。

---

## 2. 主要担忧（最可能导致 TOSEM 要求大修/补实验的点）

### 2.1 “合规性（compliance）≠ 语义正确性（correctness）”
- 评审普遍强调：消除静态告警不等同于修复功能 bug，甚至可能引入语义退化（例如删除代码/改变行为）。
- 需要进一步阐明“安全边界”：在无测试环境下，哪些规则/修复类别风险更高、需要更强 guardrail，或不宜自动修复。

### 2.2 对单一 oracle（HomeCheck）的依赖与“自证式评估”
- 主要指标来自同一个 static oracle；即便冻结配置，仍存在“以同一判定器定义目标、再用同一判定器验证”的循环风险。
- 评审希望更明确地讨论：这种评估范式的有效性边界、可能被“compliance gaming”的风险、以及如何降低风险（例如额外的轻量动态检查/多 oracle 组合/更严格回滚策略）。

### 2.3 HomeCheck 精度证据还不够“统计上坚实”
- 0 FP 的抽样结果是积极信号，但评审明确要求补充统计区间（例如 Clopper–Pearson 下界）与按规则/类别的更细粒度刻画。
- 缺少 recall（FN）评估会被接受为现实限制，但需要更清楚地阐明其影响：方法收益上界受 oracle recall 限制。

### 2.4 基线不足（归因不够强）
评审反复点名希望看到更强的对比来支撑“为何必须要这套组合”：
- **模板/确定性 auto-fix 基线**：对“模板充足”的规则，确定性修复能做到什么程度。
- **LLM-only repair 基线**：仅提供规则描述与（全文件/无你们的上下文工具）的修复效果。
- **相关范式对比**：与已有的“静态/编译/CI oracle 引导修复”类工作在概念上更清晰对齐，最好给出定量或至少结构化对比。

### 2.5 成本与效率（cost-controllable 需要证据）
- 当前扫描/静态检查时间给了，但 LLM 侧的 token/latency/每轮成本、回滚率、补丁大小分布等仍不足以支撑“cost-controllable/安全性”的强表述。

### 2.6 复现与写作细节
- 上下文抽取/合并与冲突处理策略需要更具体（重叠 group 如何合并、冲突 patch 如何处理、原子性如何保证）。
- 部分表格/片段存在截断/噪声，影响可读性与可信度。

---

## 3. 当前 TOSEM 扩展稿：哪些点已经覆盖（对齐评审关注）

结合 `-FSE-Industry2025-Learn-to-Repair-OpenHarmony-Apps/` 目录现状：

### 3.1 已覆盖/加强的点
- **范式主线与定位**：引言明确 where-to-fix/how-to-fix 分离与 precision-first 的取舍，并把 static oracle 描述为 “checkable contract”。
- **RQ 体系**：RQ1–RQ4 覆盖 oracle 可靠性、端到端迭代修复、消融与多模型。
- **语义验证**：加入分层抽样的人类语义检查与一致性统计（n=150 files，含校准语义修复率的定义）。
- **“模板 vs LLM”讨论**：在“why does it work”部分已系统讨论 template-sufficient vs LLM-needed 的概念划分。

### 3.2 仍存在的“高风险缺口”（可能被审稿人抓住）
- **RQ1 抽样覆盖偏置**：现有 RQ1 专家标注样本文件 `revision/rq1_homecheck_samples_expert.json` 中：
  - 样本量 n=200；TP=200、FP=0；
  - 但类别分布为 performance=199、security=1（security 覆盖极弱）。
  - 若论文声称“performance+security 都几乎 0 FP”，建议补充 security 分层抽样或在表述中降调并明确限制。
- **RQ1 统计区间**：对 0 FP/200 的 precision，Clopper–Pearson 95% 下界约为 **98.17%**（建议写入正文/表格）。
- **成本指标缺失**：论文目前仍缺少 LLM 侧 token/latency/迭代成本的系统统计（即使是通过日志离线统计也更可 defend）。
- **基线仍偏弱**：已有消融与多模型，但“模板/确定性修复”与“LLM-only repair（无上下文工具）”的对比不足以完全消除归因质疑。

---

## 4. 面向 TOSEM 的行动清单（按优先级）

### P0（强烈建议补，显著降低 Major Revision 风险）
1. **补齐 RQ1 的统计严谨性**
   - 在论文中报告：抽样方案、样本构成（category/severity/规则覆盖）、Clopper–Pearson 95% CI 或至少下界。
   - 明确：未测 recall 的原因与对结论的影响（收益上界受 oracle recall 限制）。
2. **补齐 security 规则的抽样覆盖**
   - 至少做一次“分层抽样确保 security 有足够样本量”的专家标注；否则把“security 高精度”弱化为“初步证据/少量样本”。
3. **补 LLM 成本与回滚/安全代理指标**
   - 建议最少报告：每轮 LLM 调用次数、每 project/每 finding 平均耗时（可用日志统计）、回滚/拒绝补丁比例。
   - 若有能力，再补：prompt 长度分布（你们日志已有 `prompt_len`）、补丁大小分布（LOC diff）、非单调回归案例数量与原因。

### P1（加分项：增强归因与“方法论贡献”的硬度）
4. **加入一个“小而硬”的模板/确定性 auto-fix 基线**
   - 选 3–5 条“模板充足、局部语法驱动”的规则（你们文中已做分类），实现保守的 deterministic fixer，并比较：
     - fixed findings、引入回归风险的 proxy（如回滚/语义抽检）。
5. **加入 LLM-only repair 基线**
   - “规则描述 + 全文件（或同等上下文预算）+ 无你们的上下文抽取/合并/差分 scaffold”的设置，展示差距。

### P2（写作与定位增强）
6. **把“oracle 自证风险”说得更硬**
   - 在 Threats/Limits 中更明确：何种风险不可避免、你们用哪些 guardrail 降低风险、未来如何引入多 oracle/轻量动态检查。
7. **清理呈现瑕疵**
   - 修复表格截断、术语统一（Static Oracle vs Linter）、数字一致性（规则数/类别数/触发规则数）。

---

## 5. 一句话结论（给内部对齐用）

评审团整体认可你把 HapRepair 上升为“静态 oracle 引导的 LLM 修复范式”的 TOSEM 叙事，但会重点追问：语义正确性与安全边界、oracle 自证风险、HomeCheck 精度的统计严谨性、以及成本/基线是否足以支撑期刊级归因与可复用结论。

