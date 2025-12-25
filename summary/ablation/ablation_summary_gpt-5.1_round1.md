Ablation Summary (gpt-5.1, round1)
==================================

Baseline (default): difflib, top_n=1, surrounding context.

| setting | init_scan_total | round1_remaining_defects | source |
| --- | --- | --- | --- |
| baseline (default) | 8626 | 505 | summary/gpt-5.1_remaining_defects.md |
| rag_top3 | 8664 | 444 | summary/ablation/gpt-5.1__ablation_rag_top3_remaining_defects.md |
| rag_top5 | 8664 | 1118 | summary/ablation/gpt-5.1__ablation_rag_top5_remaining_defects.md |
| no_rag | 8664 | 5052 | summary/ablation/gpt-5.1__ablation_no_rag_remaining_defects.md |
| diff_gpt_diff | 8664 | 476 | summary/ablation/gpt-5.1__ablation_diff_gpt_diff_remaining_defects.md |
| diff_no_diff | 8664 | 1932 | summary/ablation/gpt-5.1__ablation_diff_no_diff_remaining_defects.md |
| context_full | 8664 | 2870 | summary/ablation/gpt-5.1__ablation_context_full_remaining_defects.md |

Notes:
- Values are from the total row in each *_remaining_defects.md file.
- init_scan_total differs for baseline vs ablations because baseline uses summary/gpt-5.1_remaining_defects.md (separate run tag).

Conclusion (round 1):
- Top-3 yields the lowest remaining defects (444); Top-1 is close (505).
- Top-5 is worse than Top-1/Top-3 (1,118) but still far better than no RAG (5,052).
- GPT-diff slightly improves over difflib (476 vs 505), while no diff hurts (1,932).
- Full context is worse than surrounding context (2,870 vs 505).
