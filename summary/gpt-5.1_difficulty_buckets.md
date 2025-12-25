gpt-5.1 缺陷难度分桶统计（基于上下文/深度/合并上下文的代理指标）
====================================================================

日志来源：`logs/codelinter_openharmony/gpt-5.1`（按 CodeLinter 逐条缺陷行解析，过滤 performance/security）。

分桶规则（可用脚本参数调整）：
- `multi-defect merged context`：同一文件内，按上下文范围重叠合并后，一个上下文组包含 ≥2 条缺陷，且该缺陷需要跨函数/跨代码片段上下文（`block_ranges>1`）；
- `trivial local fixes`：不跨函数/跨片段（`block_ranges=1`），且 `context_lines≤5` 且 `brace_depth≤2`；
- `context-sensitive fixes`：其余缺陷。

**Bucketed Fix Rate（Initial → Final）**
| Bucket | Initial | Final | Fixed | Fix Rate |
| --- | --- | --- | --- | --- |
| trivial local fixes | 4268 | 64 | 4204 | 98.5% |
| context-sensitive fixes | 3531 | 113 | 3418 | 96.8% |
| multi-defect merged context | 827 | 59 | 768 | 92.9% |

**Bucket Diagnostics（Median context / depth / cross-function share）**
- Initial：Initial (round_1)
  - trivial local fixes: median_context_lines=1, median_brace_depth=1, cross_function=0.0%, multi_defect=76.7%
  - context-sensitive fixes: median_context_lines=73, median_brace_depth=3, cross_function=3.1%, multi_defect=87.2%
  - multi-defect merged context: median_context_lines=5, median_brace_depth=1, cross_function=100.0%, multi_defect=100.0%

**Final Remaining Rules (Top 15)**
- Final：After Round5
| Rule | Bucket | Final | Initial |
| --- | --- | --- | --- |
| performance/hp-arkui-no-state-var-access-in-loop | multi-defect merged context | 80 | 329 |
| performance/avoid-overusing-custom-component-check | trivial local fixes | 33 | 72 |
| performance/hp-arkui-use-onAnimationStart-for-swiper-preload | context-sensitive fixes | 27 | 245 |
| performance/hp-arkui-use-attributeUpdater-control-refresh-scope | context-sensitive fixes | 19 | 0 |
| performance/dark-color-mode-check | trivial local fixes | 14 | 14 |
| performance/foreach-index-check | context-sensitive fixes | 13 | 15 |
| performance/hp-performance-no-closures | context-sensitive fixes | 9 | 80 |
| performance/hp-performance-no-dynamic-cls-func | trivial local fixes | 8 | 17 |
| performance/hp-arkui-use-local-var-to-replace-state-var | multi-defect merged context | 6 | 417 |
| performance/monitor-invisible-area-in-image-animation | context-sensitive fixes | 4 | 5 |
| performance/hp-arkui-use-id-in-get-resource-sync-api | context-sensitive fixes | 4 | 4 |
| performance/reuse-date-instances-check | context-sensitive fixes | 3 | 17 |
| performance/hp-arkui-use-taskpool-for-web-request | trivial local fixes | 2 | 2 |
| security/no-commented-code | trivial local fixes | 2 | 429 |
| performance/hp-arkui-no-func-as-arg-for-reusable-component | context-sensitive fixes | 2 | 0 |

**Remaining Defects By Round (Buckets)**
| Bucket | Initial (round_1) | After Round1 | After Round2 | After Round3 | After Round4 | After Round5 |
| --- | --- | --- | --- | --- | --- | --- |
| trivial local fixes | 4268 | 146 | 78 | 67 | 64 | 64 |
| context-sensitive fixes | 3531 | 215 | 148 | 126 | 115 | 113 |
| multi-defect merged context | 827 | 144 | 81 | 72 | 58 | 59 |
