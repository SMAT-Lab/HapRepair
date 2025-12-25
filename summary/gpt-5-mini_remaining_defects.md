gpt-5-mini 六轮修复各项目缺陷统计
========================================

数据来源：`logs/codelinter_openharmony/gpt-5-mini` 下 `round_1`、`round_1_after_round1`、`round_2_after_round2`、`round_3_after_round3`、`round_4_after_round4`、`round_5_after_round5`、`round_6_after_round6` 的 CodeLinter 日志：优先按日志中的实际性能/安全条目数（带 @category/rule 的行）统计，若结构化解析失败再退回 `-Defects` 汇总行；
表中数值 0 表示该轮缺陷数为 0（包括：有 after_round 日志，或该轮 round_N 日志已为 0 且修复脚本提前退出未写 after_round 日志）；'-' 表示该轮没有对应日志且无法推断（通常是未检测或中途失败）。

| Project | 初始缺陷 | Round1剩余缺陷 | Round2剩余缺陷 | Round3剩余缺陷 | Round4剩余缺陷 | Round5剩余缺陷 | Round6剩余缺陷 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PullLinking | 36 | 2 | 1 | 0 | 0 | 0 | 0 |
| cameraAnimSample | 11 | 2 | 2 | 2 | 2 | 2 | 2 |
| JS_dialog_box_static | 100 | 2 | 0 | 0 | 0 | 0 | 0 |
| ace_ets_module_router1 | 99 | 4 | 0 | 0 | 0 | 0 | 0 |
| ace_ets_module_navigation1 | 89 | 0 | 0 | 0 | 0 | 0 | 0 |
| ace_ets_module_commonAttrsLayout_api11 | 85 | 1 | 1 | 1 | 1 | 1 | 1 |
| ace_ets_module_RouteManagement_api12 | 84 | 1 | 0 | 0 | 0 | 0 | 0 |
| applications_permission_manager | 80 | 20 | 0 | 0 | 0 | 0 | 0 |
| ace_ets_module_imageText_api16 | 76 | 3 | 1 | 0 | 0 | 0 | 0 |
| ace_ets_component_common_attrss_flex001 | 72 | 1 | 0 | 0 | 0 | 0 | 0 |
| audio_suite | 92 | 5 | 3 | 1 | 1 | 1 | 1 |
| applications_photos | 96 | 12 | 5 | 5 | 4 | 4 | 4 |
| applications_systemui | 74 | 20 | 12 | 5 | 3 | 3 | 3 |
| ace_ets_component_common_attrss_flex_nowear | 72 | 1 | 1 | 0 | 0 | 0 | 0 |
| acts_validator | 1357 | 37 | 24 | 2 | 2 | 2 | 2 |
| TextComponentTest | 727 | 17 | 1 | 1 | 1 | 1 | 1 |
| wifi_testapp | 544 | 80 | 63 | 50 | 47 | 46 | 35 |
| bluetoothtest | 503 | 52 | 4 | 3 | 2 | 1 | 1 |
| Info | 485 | 13 | 7 | 7 | 3 | 3 | 3 |
| ace_ets_module_nowear_waterflow | 471 | 3 | 2 | 2 | 2 | 2 | 2 |
| ace_ets_module_swiper | 457 | 47 | 33 | 28 | 25 | 23 | 18 |
| Image | 456 | 11 | 10 | 9 | 9 | 9 | 9 |
| ace_ets_module_navigation_api12 | 452 | 18 | 0 | 0 | 0 | 0 | 0 |
| ace_ets_module_scroll_nowear_api12 | 420 | 0 | 0 | 0 | 0 | 0 | 0 |
| ace_ets_module_swiper_api11 | 392 | 50 | 41 | 28 | 26 | 24 | 22 |
| ace_ets_module_imageText_api12 | 324 | 14 | 0 | 0 | 0 | 0 | 0 |
| asn1_ber | 324 | 1 | 1 | 1 | 1 | 1 | 1 |
| ohos_dfu_library | 167 | 18 | 7 | 2 | 2 | 2 | 2 |
| ohos_mail_base | 142 | 33 | 20 | 16 | 15 | 13 | 12 |
| flutter_embedding | 123 | 17 | 4 | 2 | 2 | 2 | 2 |
| HealthyPotAssistant | 56 | 7 | 2 | 2 | 2 | 2 | 2 |
| ohos_cordova | 35 | 17 | 7 | 3 | 3 | 3 | 3 |
| shopping | 117 | 22 | 19 | 17 | 16 | 16 | 15 |
| YFree_HarmonyOS | 23 | 8 | 8 | 8 | 8 | 8 | 8 |
| CanvasTest | 23 | 1 | 1 | 1 | 1 | 1 | 1 |
| 合计 | 8664 | 540 | 280 | 196 | 178 | 170 | 150 |

按规则统计（各轮剩余缺陷数，性能/安全条目）
------------------------------------------

| Rule | Initial | Round1 | Round2 | Round3 | Round4 | Round5 | Round6 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| performance/avoid-overusing-custom-component-check | 194 | 185 | 68 | 55 | 52 | 51 | 46 |
| performance/constant-property-referencing-check-in-loops | 2 | 1 | - | - | - | - | - |
| performance/dark-color-mode-check | 17 | 17 | 14 | 14 | 14 | 14 | 14 |
| performance/foreach-args-check | 249 | 5 | 5 | - | - | - | - |
| performance/foreach-index-check | 31 | 27 | 25 | 12 | 11 | 11 | 11 |
| performance/high-frequency-log-check | 22 | 3 | - | - | - | - | - |
| performance/hp-arkts-no-use-any-export-other | 38 | - | - | - | - | - | - |
| performance/hp-arkui-avoid-empty-callback | 176 | 3 | - | - | - | - | - |
| performance/hp-arkui-image-async-load | 19 | - | - | - | - | - | - |
| performance/hp-arkui-load-on-demand | 9 | - | - | - | - | - | - |
| performance/hp-arkui-no-func-as-arg-for-reusable-component | - | 4 | - | - | - | - | - |
| performance/hp-arkui-no-state-var-access-in-loop | 344 | 34 | 1 | 1 | 1 | 1 | 1 |
| performance/hp-arkui-no-stringify-in-lazyforeach-key-generator | 8 | - | - | - | - | - | - |
| performance/hp-arkui-remove-container-without-property | 763 | 22 | 5 | - | - | - | 3 |
| performance/hp-arkui-remove-redundant-nest-container | 15 | 4 | - | - | - | - | - |
| performance/hp-arkui-remove-redundant-state-var | 3429 | 115 | 14 | - | 1 | - | 2 |
| performance/hp-arkui-remove-unchanged-state-var | 4713 | 22210 | 18 | 2 | 1 | - | 2 |
| performance/hp-arkui-set-cache-count-for-lazyforeach-grid | 312 | 11 | 7 | 2 | 1 | 1 | 1 |
| performance/hp-arkui-suggest-use-effectkit-blur | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| performance/hp-arkui-use-attributeUpdater-control-refresh-scope | - | - | - | - | - | - | 1 |
| performance/hp-arkui-use-grid-layout-options | 1 | - | - | - | - | - | - |
| performance/hp-arkui-use-id-in-get-resource-sync-api | 4 | 4 | 4 | 4 | 4 | 4 | 4 |
| performance/hp-arkui-use-local-var-to-replace-state-var | 417 | 28 | 19 | 12 | 10 | 10 | 3 |
| performance/hp-arkui-use-onAnimationStart-for-swiper-preload | 252 | 70 | 49 | 36 | 30 | 28 | 24 |
| performance/hp-arkui-use-reusable-component | 426 | 54 | 45 | 28 | 28 | 27 | 16 |
| performance/hp-arkui-use-row-column-to-replace-flex | 42 | - | - | - | - | - | - |
| performance/hp-arkui-use-taskpool-for-web-request | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| performance/hp-arkui-use-transition-to-replace-animateto | 3 | - | - | - | - | - | - |
| performance/hp-arkui-use-word-break-to-replace-zero-width-space | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| performance/hp-performance-no-closures | 80 | 14 | 7 | 7 | 6 | 4 | 4 |
| performance/hp-performance-no-dynamic-cls-func | 17 | 17 | 14 | 4 | 2 | 2 | 2 |
| performance/init-list-component | 151 | 6 | 1 | - | - | - | - |
| performance/js-code-cache-by-precompile-check | 7 | 1 | 1 | 1 | 1 | 1 | 1 |
| performance/monitor-invisible-area-in-image-animation | 5 | 5 | 5 | 5 | 5 | 5 | 4 |
| performance/multiple-associations-state-var-check | 23 | 5 | 9 | 3 | 2 | 2 | 2 |
| performance/reuse-date-instances-check | 17 | 14 | 6 | 4 | 3 | 3 | 3 |
| performance/tabs-on-change-check | 6 | 6 | 3 | - | - | - | - |
| performance/waterflow-data-preload-check | 54 | 47 | 47 | - | - | - | - |
| security/no-commented-code | 431 | 12 | 1 | 1 | 1 | 1 | 1 |
| security/no-cycle | 86 | 31 | 11 | - | - | - | - |
| security/no-unsafe-hash | 1 | - | - | - | - | - | - |
