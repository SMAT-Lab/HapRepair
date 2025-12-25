qwen3-coder-plus 六轮修复各项目缺陷统计
========================================

数据来源：`logs/codelinter_openharmony/qwen3-coder-plus` 下 `round_1`、`round_1_after_round1`、`round_2_after_round2`、`round_3_after_round3`、`round_4_after_round4`、`round_5_after_round5`、`round_6_after_round6` 的 CodeLinter 日志：优先按日志中的实际性能/安全条目数（带 @category/rule 的行）统计，若结构化解析失败再退回 `-Defects` 汇总行；
表中数值 0 表示该轮缺陷数为 0（包括：有 after_round 日志，或该轮 round_N 日志已为 0 且修复脚本提前退出未写 after_round 日志）；'-' 表示该轮没有对应日志且无法推断（通常是未检测或中途失败）。

| Project | 初始缺陷 | Round1剩余缺陷 | Round2剩余缺陷 | Round3剩余缺陷 | Round4剩余缺陷 | Round5剩余缺陷 | Round6剩余缺陷 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PullLinking | 36 | 0 | 0 | 0 | 0 | 0 | - |
| cameraAnimSample | 11 | 2 | 2 | 2 | 2 | 2 | - |
| JS_dialog_box_static | 100 | 5 | 3 | 0 | 0 | 0 | - |
| ace_ets_module_router1 | 99 | 13 | 12 | 11 | 11 | 10 | - |
| ace_ets_module_navigation1 | 89 | 5 | 1 | 1 | 0 | 0 | - |
| ace_ets_module_commonAttrsLayout_api11 | 85 | 1 | 0 | 0 | 0 | 0 | - |
| ace_ets_module_RouteManagement_api12 | 84 | 14 | 14 | 14 | 14 | 13 | - |
| applications_permission_manager | 80 | 8 | 7 | 7 | 7 | 7 | - |
| ace_ets_module_imageText_api16 | 76 | 1 | 1 | 0 | 0 | 0 | - |
| ace_ets_component_common_attrss_flex001 | 72 | 0 | 0 | 0 | 0 | 0 | - |
| audio_suite | 92 | 6 | 3 | 0 | 0 | 0 | - |
| applications_photos | 96 | 4 | 2 | 2 | 2 | - | - |
| applications_systemui | 74 | 16 | 14 | 14 | 13 | 13 | - |
| ace_ets_component_common_attrss_flex_nowear | 72 | 0 | 0 | 0 | 0 | 0 | - |
| acts_validator | 1357 | 157 | 73 | 41 | 30 | 22 | - |
| TextComponentTest | 727 | 43 | 12 | 5 | 4 | 3 | - |
| wifi_testapp | 544 | 138 | 73 | 66 | 64 | 61 | - |
| bluetoothtest | 503 | 37 | 31 | 25 | 19 | 17 | - |
| Info | 485 | 23 | 11 | 7 | 6 | 5 | - |
| ace_ets_module_nowear_waterflow | 471 | 3 | 1 | 1 | 1 | 1 | - |
| ace_ets_module_swiper | 457 | 49 | 48 | 22 | 9 | 7 | - |
| Image | 456 | 145 | 125 | 125 | 65 | - | - |
| ace_ets_module_navigation_api12 | 452 | 1 | 0 | 0 | 0 | 0 | - |
| ace_ets_module_scroll_nowear_api12 | 420 | 1 | 1 | 1 | 1 | 0 | - |
| ace_ets_module_swiper_api11 | 392 | 46 | 16 | 15 | 10 | 8 | - |
| ace_ets_module_imageText_api12 | 324 | 35 | 35 | 8 | 6 | 6 | - |
| asn1_ber | 324 | 57 | 57 | 57 | 57 | 57 | - |
| ohos_dfu_library | 167 | 19 | 14 | 13 | 4 | 4 | - |
| ohos_mail_base | 142 | 27 | 17 | 15 | 11 | 11 | - |
| flutter_embedding | 123 | 33 | 17 | 12 | 1 | - | - |
| HealthyPotAssistant | 56 | 6 | 4 | 3 | 3 | 3 | - |
| ohos_cordova | 35 | 12 | 8 | 5 | 3 | 3 | - |
| shopping | 117 | 20 | 15 | 14 | 13 | - | - |
| YFree_HarmonyOS | 23 | 3 | 3 | 3 | 3 | - | - |
| CanvasTest | 23 | 0 | 0 | 0 | 0 | 0 | - |
| 合计 | 8664 | 930 | 620 | 489 | 359 | 253 | 0 |

按规则统计（各轮剩余缺陷数，性能/安全条目）
------------------------------------------

| Rule | Initial | Round1 | Round2 | Round3 | Round4 | Round5 | Round6 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| performance/avoid-overusing-custom-component-check | 72 | 19 | 11 | 9 | 8 | 4 | - |
| performance/constant-property-referencing-check-in-loops | 2 | 3 | 1 | - | - | - | - |
| performance/dark-color-mode-check | 14 | 14 | 14 | 14 | 14 | 10 | - |
| performance/foreach-args-check | 162 | - | - | - | - | - | - |
| performance/foreach-index-check | 15 | 22 | 20 | 18 | 19 | 7 | - |
| performance/high-frequency-log-check | 2 | - | - | 3 | - | - | - |
| performance/hp-arkts-no-use-any-export-other | 30 | - | - | - | - | - | - |
| performance/hp-arkui-avoid-empty-callback | 171 | 150 | 99 | 58 | 35 | 24 | - |
| performance/hp-arkui-image-async-load | 5 | - | - | - | - | - | - |
| performance/hp-arkui-load-on-demand | 1 | - | - | - | - | - | - |
| performance/hp-arkui-no-func-as-arg-for-reusable-component | - | 19 | 21 | 19 | 11 | 8 | - |
| performance/hp-arkui-no-state-var-access-in-loop | 329 | 204 | 188 | 175 | 111 | 49 | - |
| performance/hp-arkui-no-stringify-in-lazyforeach-key-generator | 6 | - | - | - | - | - | - |
| performance/hp-arkui-remove-container-without-property | 623 | 1 | - | - | - | - | - |
| performance/hp-arkui-remove-redundant-nest-container | 14 | 4 | 1 | - | - | - | - |
| performance/hp-arkui-remove-redundant-state-var | 2490 | 74 | 12 | - | - | - | - |
| performance/hp-arkui-remove-unchanged-state-var | 2695 | 104 | 20 | 1 | 2 | 1 | - |
| performance/hp-arkui-set-cache-count-for-lazyforeach-grid | 275 | 2 | - | - | - | - | - |
| performance/hp-arkui-suggest-use-effectkit-blur | 2 | 2 | 2 | 2 | 2 | 2 | - |
| performance/hp-arkui-use-attributeUpdater-control-refresh-scope | - | 5 | 7 | 7 | 7 | 7 | - |
| performance/hp-arkui-use-grid-layout-options | 1 | - | - | - | - | - | - |
| performance/hp-arkui-use-id-in-get-resource-sync-api | 4 | 4 | 4 | 4 | 4 | 4 | - |
| performance/hp-arkui-use-local-var-to-replace-state-var | 417 | 109 | 109 | 109 | 109 | 109 | - |
| performance/hp-arkui-use-onAnimationStart-for-swiper-preload | 245 | 13 | 8 | 5 | 1 | 1 | - |
| performance/hp-arkui-use-reusable-component | 371 | 52 | 26 | 5 | 4 | 1 | - |
| performance/hp-arkui-use-row-column-to-replace-flex | 40 | 1 | - | - | - | - | - |
| performance/hp-arkui-use-taskpool-for-web-request | 2 | 2 | 2 | 2 | 2 | - | - |
| performance/hp-arkui-use-transition-to-replace-animateto | 2 | - | - | - | - | - | - |
| performance/hp-arkui-use-word-break-to-replace-zero-width-space | 1 | 1 | 1 | 1 | 1 | 1 | - |
| performance/hp-performance-no-closures | 80 | 36 | 17 | 11 | 7 | 6 | - |
| performance/hp-performance-no-dynamic-cls-func | 17 | 14 | 14 | 13 | 12 | 12 | - |
| performance/init-list-component | 19 | - | - | - | - | - | - |
| performance/js-code-cache-by-precompile-check | 4 | 2 | 2 | 2 | 1 | 1 | - |
| performance/monitor-invisible-area-in-image-animation | 5 | 3 | 2 | 2 | 2 | - | - |
| performance/multiple-associations-state-var-check | 17 | 2 | - | - | - | - | - |
| performance/reuse-date-instances-check | 17 | 12 | 6 | 4 | 4 | 4 | - |
| security/no-commented-code | 429 | 2 | 1 | 1 | 1 | - | - |
| security/no-cycle | 84 | 54 | 32 | 24 | 2 | 2 | - |
| security/no-unsafe-hash | 1 | - | - | - | - | - | - |
