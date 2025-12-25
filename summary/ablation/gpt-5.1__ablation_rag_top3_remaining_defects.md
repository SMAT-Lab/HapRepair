gpt-5.1__ablation_rag_top3 六轮修复各项目缺陷统计
========================================

数据来源：`logs/codelinter_openharmony/gpt-5.1__ablation_rag_top3` 下 `round_1`、`round_1_after_round1`、`round_2_after_round2`、`round_3_after_round3`、`round_4_after_round4`、`round_5_after_round5`、`round_6_after_round6` 的 CodeLinter 日志：优先按日志中的实际性能/安全条目数（带 @category/rule 的行）统计，若结构化解析失败再退回 `-Defects` 汇总行；
“初始缺陷(基线)”取自 `revision/target_projects_haprepair.json` 中的性能/安全缺陷数（实验配置基线，稳定且与模型无关）；
“初始缺陷(扫描)”取自本次运行 `round_1/<project>.log` 的实际扫描结果（可能受工具版本/环境影响）；
表中数值 0 表示该轮缺陷数为 0（包括：有 after_round 日志，或该轮 round_N 日志已为 0 且修复脚本提前退出未写 after_round 日志）；'-' 表示该轮没有对应日志且无法推断（通常是未检测或中途失败）。

| Project | 初始缺陷(基线) | 初始缺陷(扫描) | Round1剩余缺陷 | Round2剩余缺陷 | Round3剩余缺陷 | Round4剩余缺陷 | Round5剩余缺陷 | Round6剩余缺陷 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PullLinking | 36 | 36 | 0 | - | - | - | - | - |
| cameraAnimSample | 11 | 11 | 2 | - | - | - | - | - |
| JS_dialog_box_static | 100 | 100 | 0 | - | - | - | - | - |
| ace_ets_module_router1 | 99 | 99 | 4 | - | - | - | - | - |
| ace_ets_module_navigation1 | 89 | 89 | 1 | - | - | - | - | - |
| ace_ets_module_commonAttrsLayout_api11 | 85 | 85 | 1 | - | - | - | - | - |
| ace_ets_module_RouteManagement_api12 | 84 | 84 | 2 | - | - | - | - | - |
| applications_permission_manager | 80 | 80 | 8 | - | - | - | - | - |
| ace_ets_module_imageText_api16 | 76 | 76 | 0 | - | - | - | - | - |
| ace_ets_component_common_attrss_flex001 | 72 | 72 | 0 | - | - | - | - | - |
| audio_suite | 92 | 92 | 6 | - | - | - | - | - |
| applications_photos | 96 | 96 | 7 | - | - | - | - | - |
| applications_systemui | 74 | 74 | 17 | - | - | - | - | - |
| ace_ets_component_common_attrss_flex_nowear | 72 | 72 | 0 | - | - | - | - | - |
| acts_validator | 1357 | 1357 | 24 | - | - | - | - | - |
| TextComponentTest | 727 | 727 | 1 | - | - | - | - | - |
| wifi_testapp | 544 | 544 | 52 | - | - | - | - | - |
| bluetoothtest | 503 | 503 | 24 | - | - | - | - | - |
| Info | 485 | 485 | 8 | - | - | - | - | - |
| ace_ets_module_nowear_waterflow | 471 | 471 | 0 | - | - | - | - | - |
| ace_ets_module_swiper | 457 | 457 | 36 | - | - | - | - | - |
| Image | 456 | 456 | 127 | - | - | - | - | - |
| ace_ets_module_navigation_api12 | 452 | 452 | 0 | - | - | - | - | - |
| ace_ets_module_scroll_nowear_api12 | 420 | 420 | 0 | - | - | - | - | - |
| ace_ets_module_swiper_api11 | 392 | 392 | 30 | - | - | - | - | - |
| ace_ets_module_imageText_api12 | 324 | 324 | 1 | - | - | - | - | - |
| asn1_ber | 324 | 324 | 0 | - | - | - | - | - |
| ohos_dfu_library | 167 | 167 | 15 | - | - | - | - | - |
| ohos_mail_base | 142 | 142 | 16 | - | - | - | - | - |
| flutter_embedding | 123 | 123 | 20 | - | - | - | - | - |
| HealthyPotAssistant | 56 | 56 | 7 | - | - | - | - | - |
| ohos_cordova | 35 | 35 | 8 | - | - | - | - | - |
| shopping | 117 | 117 | 20 | - | - | - | - | - |
| YFree_HarmonyOS | 23 | 23 | 6 | - | - | - | - | - |
| CanvasTest | 23 | 23 | 1 | - | - | - | - | - |
| 合计 | 8664 | 8664 | 444 | 0 | 0 | 0 | 0 | 0 |

按规则统计（各轮剩余缺陷数，性能/安全条目）
------------------------------------------

| Rule | Initial | Round1 | Round2 | Round3 | Round4 | Round5 | Round6 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| performance/avoid-overusing-custom-component-check | 72 | 56 | - | - | - | - | - |
| performance/constant-property-referencing-check-in-loops | 2 | 1 | - | - | - | - | - |
| performance/dark-color-mode-check | 14 | 14 | - | - | - | - | - |
| performance/foreach-args-check | 162 | - | - | - | - | - | - |
| performance/foreach-index-check | 15 | 13 | - | - | - | - | - |
| performance/high-frequency-log-check | 2 | - | - | - | - | - | - |
| performance/hp-arkts-no-use-any-export-other | 30 | - | - | - | - | - | - |
| performance/hp-arkui-avoid-empty-callback | 171 | 1 | - | - | - | - | - |
| performance/hp-arkui-image-async-load | 5 | - | - | - | - | - | - |
| performance/hp-arkui-load-on-demand | 1 | - | - | - | - | - | - |
| performance/hp-arkui-no-func-as-arg-for-reusable-component | - | 6 | - | - | - | - | - |
| performance/hp-arkui-no-state-var-access-in-loop | 329 | 141 | - | - | - | - | - |
| performance/hp-arkui-no-stringify-in-lazyforeach-key-generator | 6 | - | - | - | - | - | - |
| performance/hp-arkui-remove-container-without-property | 623 | 1 | - | - | - | - | - |
| performance/hp-arkui-remove-redundant-nest-container | 14 | 4 | - | - | - | - | - |
| performance/hp-arkui-remove-redundant-state-var | 2490 | 15 | - | - | - | - | - |
| performance/hp-arkui-remove-unchanged-state-var | 2695 | 16 | - | - | - | - | - |
| performance/hp-arkui-set-cache-count-for-lazyforeach-grid | 275 | 1 | - | - | - | - | - |
| performance/hp-arkui-suggest-use-effectkit-blur | 2 | 2 | - | - | - | - | - |
| performance/hp-arkui-use-attributeUpdater-control-refresh-scope | - | 3 | - | - | - | - | - |
| performance/hp-arkui-use-grid-layout-options | 1 | 1 | - | - | - | - | - |
| performance/hp-arkui-use-id-in-get-resource-sync-api | 4 | - | - | - | - | - | - |
| performance/hp-arkui-use-local-var-to-replace-state-var | 417 | 6 | - | - | - | - | - |
| performance/hp-arkui-use-onAnimationStart-for-swiper-preload | 245 | 49 | - | - | - | - | - |
| performance/hp-arkui-use-reusable-component | 371 | 11 | - | - | - | - | - |
| performance/hp-arkui-use-row-column-to-replace-flex | 40 | - | - | - | - | - | - |
| performance/hp-arkui-use-taskpool-for-web-request | 2 | 2 | - | - | - | - | - |
| performance/hp-arkui-use-transition-to-replace-animateto | 2 | - | - | - | - | - | - |
| performance/hp-arkui-use-word-break-to-replace-zero-width-space | 1 | 1 | - | - | - | - | - |
| performance/hp-performance-no-closures | 80 | 33 | - | - | - | - | - |
| performance/hp-performance-no-dynamic-cls-func | 17 | 14 | - | - | - | - | - |
| performance/init-list-component | 19 | - | - | - | - | - | - |
| performance/js-code-cache-by-precompile-check | 4 | 2 | - | - | - | - | - |
| performance/monitor-invisible-area-in-image-animation | 5 | 5 | - | - | - | - | - |
| performance/multiple-associations-state-var-check | 17 | 6 | - | - | - | - | - |
| performance/reuse-date-instances-check | 17 | 8 | - | - | - | - | - |
| security/no-commented-code | 429 | 3 | - | - | - | - | - |
| security/no-cycle | 84 | 29 | - | - | - | - | - |
| security/no-unsafe-hash | 1 | - | - | - | - | - | - |
