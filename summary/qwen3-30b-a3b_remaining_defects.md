qwen3-30b-a3b 六轮修复各项目缺陷统计
========================================

数据来源：`logs/codelinter_openharmony/qwen3-30b-a3b` 下 `round_1`、`round_1_after_round1`、`round_2_after_round2`、`round_3_after_round3`、`round_4_after_round4`、`round_5_after_round5`、`round_6_after_round6` 的 CodeLinter 日志：优先按日志中的实际性能/安全条目数（带 @category/rule 的行）统计，若结构化解析失败再退回 `-Defects` 汇总行；
“初始缺陷(基线)”取自 `revision/target_projects_haprepair.json` 中的性能/安全缺陷数（实验配置基线，稳定且与模型无关）；
“初始缺陷(扫描)”取自本次运行 `round_1/<project>.log` 的实际扫描结果（可能受工具版本/环境影响）；
表中数值 0 表示该轮缺陷数为 0（包括：有 after_round 日志，或该轮 round_N 日志已为 0 且修复脚本提前退出未写 after_round 日志）；'-' 表示该轮没有对应日志且无法推断（通常是未检测或中途失败）。

| Project | 初始缺陷(基线) | 初始缺陷(扫描) | Round1剩余缺陷 | Round2剩余缺陷 | Round3剩余缺陷 | Round4剩余缺陷 | Round5剩余缺陷 | Round6剩余缺陷 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PullLinking | 36 | 36 | 0 | 0 | 0 | 0 | 0 | - |
| cameraAnimSample | 11 | 11 | 7 | 4 | 4 | 4 | 4 | - |
| JS_dialog_box_static | 100 | 100 | 4 | 4 | 4 | 4 | 4 | - |
| ace_ets_module_router1 | 99 | 99 | 18 | 9 | 7 | 7 | 7 | - |
| ace_ets_module_navigation1 | 89 | 89 | 50 | 12 | 12 | 12 | 12 | - |
| ace_ets_module_commonAttrsLayout_api11 | 85 | 85 | 18 | 8 | 2 | 2 | 0 | - |
| ace_ets_module_RouteManagement_api12 | 84 | 84 | 14 | 13 | 11 | - | 13 | - |
| applications_permission_manager | 80 | 80 | 29 | 10 | 3 | 3 | 3 | - |
| ace_ets_module_imageText_api16 | 76 | 76 | 19 | 2 | 1 | 0 | 0 | - |
| ace_ets_component_common_attrss_flex001 | 72 | 72 | 6 | 6 | 0 | 0 | 0 | - |
| audio_suite | 92 | 92 | 40 | 32 | 19 | 13 | 13 | - |
| applications_photos | 96 | 96 | 30 | 22 | 22 | 21 | 19 | - |
| applications_systemui | 74 | 74 | 48 | 37 | 27 | 26 | 22 | - |
| ace_ets_component_common_attrss_flex_nowear | 72 | 72 | 0 | 0 | 0 | 0 | 0 | - |
| acts_validator | 1357 | 1357 | 792 | 548 | 330 | 226 | 195 | - |
| TextComponentTest | 727 | 727 | 400 | 284 | 262 | 236 | 208 | - |
| wifi_testapp | 544 | 544 | 384 | 245 | 157 | 130 | 119 | - |
| bluetoothtest | 503 | 503 | 289 | 228 | 176 | 130 | 126 | - |
| Info | 485 | 485 | 240 | 205 | - | 229 | 194 | - |
| ace_ets_module_nowear_waterflow | 471 | 471 | 8 | 4 | 1 | 1 | 1 | - |
| ace_ets_module_swiper | 457 | 457 | 95 | 31 | 26 | 26 | 26 | - |
| Image | 456 | 456 | 246 | 177 | 177 | 168 | 162 | - |
| ace_ets_module_navigation_api12 | 452 | 452 | 42 | 33 | 31 | 20 | 20 | - |
| ace_ets_module_scroll_nowear_api12 | 420 | 420 | 11 | 9 | 9 | 9 | 0 | - |
| ace_ets_module_swiper_api11 | 392 | 392 | 68 | 38 | 28 | 18 | 18 | - |
| ace_ets_module_imageText_api12 | 324 | 324 | 206 | 134 | 99 | 88 | 76 | - |
| asn1_ber | 324 | 324 | 0 | 0 | 0 | 0 | 0 | - |
| ohos_dfu_library | 167 | 167 | 29 | 25 | 23 | 19 | 13 | - |
| ohos_mail_base | 142 | 142 | 24 | 18 | 18 | 17 | 16 | - |
| flutter_embedding | 123 | 123 | 33 | 15 | - | - | 46 | - |
| HealthyPotAssistant | 56 | 56 | 11 | 10 | 9 | 4 | 3 | - |
| ohos_cordova | 35 | 35 | 16 | 9 | 9 | 7 | 7 | - |
| shopping | 117 | 117 | 19 | 14 | 14 | 13 | 13 | - |
| YFree_HarmonyOS | 23 | 23 | 8 | 8 | 8 | 8 | 8 | - |
| CanvasTest | 23 | 23 | 3 | 3 | 3 | 3 | 3 | - |
| 合计 | 8664 | 8664 | 3207 | 2197 | 1492 | 1444 | 1351 | 0 |

按规则统计（各轮剩余缺陷数，性能/安全条目）
------------------------------------------

| Rule | Initial | Round1 | Round2 | Round3 | Round4 | Round5 | Round6 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| performance/avoid-overusing-custom-component-check | 72 | 69 | 67 | 59 | 62 | 56 | - |
| performance/constant-property-referencing-check-in-loops | 2 | 1 | 1 | 1 | 1 | 2 | - |
| performance/dark-color-mode-check | 14 | 14 | 14 | 13 | 13 | 14 | - |
| performance/foreach-args-check | 162 | 1 | 1 | - | - | - | - |
| performance/foreach-index-check | 15 | 15 | 15 | 11 | 12 | 12 | - |
| performance/high-frequency-log-check | 2 | - | - | - | - | - | - |
| performance/hp-arkts-no-use-any-export-other | 30 | - | - | - | - | - | - |
| performance/hp-arkui-avoid-empty-callback | 171 | 111 | 43 | 25 | 27 | 19 | - |
| performance/hp-arkui-image-async-load | 5 | 2 | 2 | 2 | 2 | 2 | - |
| performance/hp-arkui-load-on-demand | 1 | - | - | - | - | - | - |
| performance/hp-arkui-no-func-as-arg-for-reusable-component | - | 8 | 6 | 3 | 1 | 1 | - |
| performance/hp-arkui-no-state-var-access-in-loop | 329 | 145 | 129 | 127 | 108 | 114 | - |
| performance/hp-arkui-no-stringify-in-lazyforeach-key-generator | 6 | 3 | 2 | 2 | 1 | 1 | - |
| performance/hp-arkui-remove-container-without-property | 623 | 17 | 17 | 6 | - | - | - |
| performance/hp-arkui-remove-redundant-nest-container | 14 | 4 | - | - | - | - | - |
| performance/hp-arkui-remove-redundant-state-var | 2490 | 1257 | 856 | 553 | 548 | 501 | - |
| performance/hp-arkui-remove-unchanged-state-var | 2695 | 1220 | 814 | 503 | 505 | 446 | - |
| performance/hp-arkui-set-cache-count-for-lazyforeach-grid | 275 | 4 | 1 | 1 | 1 | 1 | - |
| performance/hp-arkui-suggest-use-effectkit-blur | 2 | 2 | 2 | 2 | 2 | 2 | - |
| performance/hp-arkui-use-attributeUpdater-control-refresh-scope | - | 10 | 30 | 31 | 31 | 31 | - |
| performance/hp-arkui-use-grid-layout-options | 1 | 1 | 1 | 1 | 1 | 1 | - |
| performance/hp-arkui-use-id-in-get-resource-sync-api | 4 | 3 | 2 | 2 | 2 | 2 | - |
| performance/hp-arkui-use-local-var-to-replace-state-var | 417 | 60 | 37 | 34 | 27 | 27 | - |
| performance/hp-arkui-use-onAnimationStart-for-swiper-preload | 245 | 15 | 7 | 1 | 1 | 1 | - |
| performance/hp-arkui-use-reusable-component | 371 | 79 | 25 | 16 | 11 | 11 | - |
| performance/hp-arkui-use-row-column-to-replace-flex | 40 | - | - | - | - | - | - |
| performance/hp-arkui-use-taskpool-for-web-request | 2 | 2 | 1 | 1 | 1 | 1 | - |
| performance/hp-arkui-use-transition-to-replace-animateto | 2 | 1 | 1 | 1 | 1 | 1 | - |
| performance/hp-arkui-use-word-break-to-replace-zero-width-space | 1 | 1 | 1 | 1 | 1 | 1 | - |
| performance/hp-performance-no-closures | 80 | 62 | 59 | 56 | 51 | 50 | - |
| performance/hp-performance-no-dynamic-cls-func | 17 | 16 | 14 | 9 | 9 | 6 | - |
| performance/init-list-component | 19 | - | - | - | - | - | - |
| performance/js-code-cache-by-precompile-check | 4 | 2 | 2 | 2 | 2 | 2 | - |
| performance/monitor-invisible-area-in-image-animation | 5 | 5 | 5 | 5 | 5 | 5 | - |
| performance/multiple-associations-state-var-check | 17 | 5 | 2 | 1 | 1 | 1 | - |
| performance/reuse-date-instances-check | 17 | 15 | 12 | 7 | 6 | 6 | - |
| security/no-commented-code | 429 | 3 | 2 | 2 | 2 | 2 | - |
| security/no-cycle | 84 | 54 | 26 | 14 | 9 | 32 | - |
| security/no-unsafe-hash | 1 | - | - | - | - | - | - |
