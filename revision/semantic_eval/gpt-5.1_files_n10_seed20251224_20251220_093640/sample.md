# File-level repair semantic-eval sample (gpt-5.1)

- n_files=10, seed=20251224, max_round=5
- max_lines=600, max_chars=60000

## 73e600229b5b
- project: `ohos_mail_base`
- round_fixed: `1` (early)
- file: `lib/src/main/ets/format/msg.ts`
- fixed_findings: `27` (both)

## 901752306d6e
- project: `shopping`
- round_fixed: `1` (early)
- file: `entry/src/main/ets/pages/Index.ets`
- fixed_findings: `17` (both)

## 257d893d639a
- project: `Image`
- round_fixed: `2` (early)
- file: `entry/src/main/ets/pages/ImageRotate/ImageRotateExample001.ets`
- fixed_findings: `1` (performance_only)

## 8897d11a5c0f
- project: `TextComponentTest`
- round_fixed: `1` (early)
- file: `entry/src/main/ets/pages/01-StyledString/02-decoration/01-textDecorationType.ets`
- fixed_findings: `1` (performance_only)

## ef5d5490601b
- project: `ohos_dfu_library`
- round_fixed: `2` (early)
- file: `DfuLibrary/src/main/ets/dfu/LegacyDfuImpl.ts`
- fixed_findings: `1` (security_only)

## 9fdaddd60fbc
- project: `ohos_mail_base`
- round_fixed: `1` (early)
- file: `lib/src/main/ets/format/rtf.ts`
- fixed_findings: `1` (security_only)

## d5b7a20cf735
- project: `ace_ets_module_swiper`
- round_fixed: `4` (late)
- file: `entry/src/main/ets/MainAbility/pages/Swiper/swiper2/siwpe_leftscroll.ets`
- fixed_findings: `1` (performance_only)

## a94a8dddc1e0
- project: `ace_ets_module_swiper_api11`
- round_fixed: `3` (late)
- file: `entry/src/main/ets/MainAbility/pages/Swiper/swiperUI/swiperNextPrevMargin59.ets`
- fixed_findings: `1` (performance_only)

## 2af0d1145566
- project: `flutter_embedding`
- round_fixed: `3` (late)
- file: `flutter/src/main/ets/embedding/engine/FlutterEngineConnectionRegistry.ets`
- fixed_findings: `1` (security_only)

## f763287a8d2b
- project: `flutter_embedding`
- round_fixed: `3` (late)
- file: `flutter/src/main/ets/embedding/engine/FlutterEngineGroup.ets`
- fixed_findings: `1` (security_only)
