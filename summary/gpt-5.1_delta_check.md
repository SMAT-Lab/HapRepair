gpt-5.1 Delta Check（代码变化量反博弈统计）
======================================

目的：针对主实验的 8,428 个净减少（net fixed）oracle 发现，补充基于代码变化量的统计，用于排除“通过大规模删除代码来降低静态发现数”的潜在博弈行为。

数据与口径（可复现）：
- 基线发现：`logs/codelinter_openharmony/deepseek-chat/round_1/*.log`（用于获得完整基线定位；总数以 `revision/target_projects_haprepair.json` 的 8,664 为准）
- 最终发现：`logs/codelinter_openharmony/gpt-5.1/round_5_after_round5/*.log`（若缺失则回退到最晚可用的 `round_r_after_roundr`；通常是项目早期清零后不再生成后续日志）
- 代码差分：对每个“发现净减少”的文件，比较 `revision/target_projects_haprepair.json` 中 `root_path`（原始快照）与 `revision/fixed_projects/gpt-5.1/round_5/<project>/...`（最终修复快照）的文件级 diff；并按该文件贡献的“resolved findings（decrease）”数量进行加权统计。

复现命令：
- `python3 revision/code/delta_check_summarize.py --allow-missing-final-logs`

输出：
- 结构化统计：`summary/gpt-5.1_delta_check.json`
- 论文 LaTeX 表：`-FSE-Industry2025-Learn-to-Repair-OpenHarmony-Apps/delta_check_table.tex`

关键结果（finding-weighted）：
- Net fixed（baseline minus final）：8,428
- Resolved（decrease，用于加权）：8,451；Introduced（increase）：23
- 删除行数分布：median=0；P95=16
- 仅 2/1,578 个“有 resolved 的文件”出现 >=100 行删除；它们都来自 `Image` 项目（两文件合计贡献 101 个 resolved findings，因此在 finding-weighted 口径下占 1.20%）
- 保守过滤（去除所有“可疑大删除”相关 resolved cases）后：96.11%（8,327/8,664）
