# CodeLinter 高缺陷规则中“难以确定性自动修复”的候选（用于 TOSEM 扩展版讨论）

数据来源：`logs/codelinter_openharmony/gpt-5.1/round_1/*.log`（共匹配到 8628 条带 `@performance/...` / `@security/...` 的缺陷行）。

## 1) Round_1 扫描：规则频次 Top（按缺陷条数）

> 说明：这里仅用于“选 case”。我们不做“多少条规则可自动修”的总体统计；只把已经有确定性 auto-fix 的规则标出来，后续 case 选择时避开。

| Rule | Count | 备注 |
| --- | --- | --- |
| `performance/hp-arkui-remove-unchanged-state-var` | 2690 | **难以确定性修复**（需全局数据流/引用分析） |
| `performance/hp-arkui-remove-redundant-state-var` | 2482 | **难以确定性修复**（需 UI 绑定/使用点分析） |
| `performance/hp-arkui-remove-container-without-property` | 623 | 已有确定性 auto-fix（结构化删容器） |
| `security/no-commented-code` | 429 | 已有确定性 auto-fix（删注释行） |
| `performance/hp-arkui-use-local-var-to-replace-state-var` | 417 | 有启发式 auto-fix 但默认关闭（行为敏感） |
| `performance/hp-arkui-use-reusable-component` | 371 | **难以确定性修复**（组件重构/接口调整） |
| `performance/hp-arkui-no-state-var-access-in-loop` | 329 | **难以确定性修复**（循环外缓存 + 同步写回语义） |
| `performance/hp-arkui-set-cache-count-for-lazyforeach-grid` | 275 | 已有确定性 auto-fix（插入 `.cachedCount(4)`） |
| `performance/hp-arkui-use-onAnimationStart-for-swiper-preload` | 245 | **难以确定性修复**（需要理解 Swiper 结构与预加载逻辑） |
| `performance/hp-arkui-avoid-empty-callback` | 169 | 可能可规则化（需确认典型模式） |
| `performance/foreach-args-check` | 162 | 已有确定性 auto-fix（补 `keyGenerator`） |
| `security/no-cycle` | 84 | **难以确定性修复**（需要架构级重构/依赖解环） |
| `performance/hp-performance-no-closures` | 80 | **难以确定性修复**（闭包消除可能引入语义变化） |
| `performance/avoid-overusing-custom-component-check` | 72 | **难以确定性修复**（需要把自定义组件改写为 `@Builder`） |
| `performance/hp-arkui-use-row-column-to-replace-flex` | 40 | 已有确定性 auto-fix（结构替换） |

## 2) 建议挑选的“高缺陷 + 难以确定性修复”规则（case 候选）

下面每条给出：为什么很难用确定性模板安全修、以及 2 个来自 Round_1 扫描日志的代表性位置（便于写 paper 的例子/图）。

## 2.0 TOSEM 扩展版：最终选择的 3 条 case（组件/Swiper 重构类，用于说明为何需要 LLM）

我们决定用更“重构型”的规则来支撑论文核心论点之一：这些缺陷往往需要跨函数/跨组件的语义保持改写与设计取舍，难以用固定模板或简单 AST 重写覆盖，因此需要 LLM 进行上下文理解与生成式重构（同时再配合编译/测试/静态检查做验证）。

| Rule | Initial | Round1 | Round2 | Round3 | Round4 | Round5 |
| --- | --- | --- | --- | --- | --- | --- |
| `performance/hp-arkui-use-reusable-component` | 371 | 26 | 7 | 2 | 1 | 2 |
| `performance/hp-arkui-use-onAnimationStart-for-swiper-preload` | 245 | 53 | 38 | 32 | 27 | 27 |
| `performance/avoid-overusing-custom-component-check` | 72 | 51 | 38 | 34 | 33 | 33 |

> 这些数值来自 `summary/gpt-5.1_remaining_defects.md` 的“按规则统计（各轮剩余缺陷数）”表。

### 2.1 `performance/hp-arkui-remove-unchanged-state-var`（2690）

- 难点：需要证明该 `@State`（或同等状态变量）不会被修改，且删除后不会影响 UI 绑定、序列化、反射/装饰器行为；确定性修复需要跨作用域/跨方法数据流分析。
- 例子：
  - `CanvasTest:entry/src/main/ets/pages/CanvasPage.ets:396:3`（Remove unchanged state variables）
  - `CanvasTest:entry/src/main/ets/pages/Index.ets:65:3`（Remove unchanged state variables）

### 2.2 `performance/hp-arkui-remove-redundant-state-var`（2482）

- 难点：需要判断状态变量是否“确实未参与 UI 渲染/绑定”，且没有间接使用（例如作为参数传入、被 `@Builder`/组件引用、或用于条件分支触发重绘）；确定性修复要做较强的使用点追踪。
- 例子：
  - `CanvasTest:entry/src/main/ets/pages/CanvasPage.ets:395:3`（Remove state variables that are not associated with a UI component）
  - `CanvasTest:entry/src/main/ets/pages/CanvasPage.ets:396:3`（Remove state variables that are not associated with a UI component）

### 2.3 `performance/hp-arkui-no-state-var-access-in-loop`（329）

- 难点：表面上是“循环外缓存 state 读”，但需要处理：state 是否在循环内被写、是否依赖每次迭代的最新值、以及缓存变量与 state 的同步写回点；错误的模板化改写容易引入逻辑 bug。
- 例子：
  - `Image:entry/src/main/ets/pages/ImageParallelization/ImageCallBack.ets:113:15`（Avoid frequent state variable reads inside loop logic）
  - `Image:entry/src/main/ets/pages/ImageParallelization/ImageCallBack.ets:118:17`（Avoid frequent state variable reads inside loop logic）

### 2.4 `performance/hp-arkui-use-reusable-component`（371）

- 难点：通常需要把复杂 UI 片段抽成 reusable component（或 builder/子组件），涉及 props/参数设计、状态提升、样式/布局继承；很难用固定模板直接“挪代码”而不破坏编译/行为。
- 例子：
  - `ace_ets_module_commonAttrsLayout_api11:entry/src/main/ets/MainAbility/pages/comSizeAbility/comSizeAbilityTest_28.ets:60:13`（Use reusable components to define complex components whenever possible）
  - `ace_ets_module_nowear_waterflow:entry/src/main/ets/MainAbility/pages/WaterFlow/cachedShowFlow.ets:54:15`（Use reusable components to define complex components whenever possible）

### 2.5 `performance/hp-arkui-use-onAnimationStart-for-swiper-preload`（245）

- 难点：需要理解 Swiper 的数据源、预加载触发逻辑、以及回调接线位置；确定性修复不仅是“加一个回调”，往往还要改现有回调/状态更新时机。
- 例子：
  - `ace_ets_module_commonAttrsLayout_api11:entry/src/main/ets/MainAbility/pages/comSizeAbility/comSizeAbilityTest_28.ets:58:9`（Use the swiper preloading mechanism with the OnAnimationStart callback）
  - `ace_ets_module_swiper:entry/src/main/ets/MainAbility/pages/Swiper/maintainPosition.ets:48:7`（Use the swiper preloading mechanism with the OnAnimationStart callback）

### 2.6 `performance/avoid-overusing-custom-component-check`（72）

- 难点：需要把自定义组件“重写成 @Builder”，牵涉到参数、状态捕获、slot/children 结构，且对可读性与复用边界有设计取舍。
- 例子：
  - `CanvasTest:entry/src/main/ets/pages/XcomponentPage.ets:5:8`（Preferentially use the @Builder method instead of custom components.）
  - `HealthyPotAssistant:entry/src/main/ets/MainAbility/components/MTopBar.ets:5:15`（Preferentially use the @Builder method instead of custom components.）

## 3) 这段内容在论文里怎么用（建议写法）

- 放在“为何需要 LLM（而不仅是规则/模板/AST 重写）”的论证里：展示高频规则中一部分属于“语义/架构级重构”，并给出具体扫描例子（上面的 3 条就够）。
- 同时说明工程化取舍：对少量结构性、低风险、确定性可做的规则（删冗余容器、补参数、插入链式调用），我们会剥离成 deterministic auto-fix（减少 LLM 成本/降低误改风险）；而对上述重构型规则，主要依赖 LLM + 验证闭环。
