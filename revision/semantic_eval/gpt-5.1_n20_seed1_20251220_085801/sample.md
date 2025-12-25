# Repair semantic-eval sample (gpt-5.1)

- n=20, seed=1, max_round=1

## 345f1cbb3836
- project: `Image`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-redundant-state-var` (suggestion)
- file: `entry/src/main/ets/pages/ImageHdrExamples/ImageHdrExample006.ets`:41:3
- patch: shape=`modify`, changed_lines=10

### Before (local context)
```
    35  @Component
    36  struct DarkModeTest {
    37    @State modifier: ImageModifier = new ImageModifier()
    38    @State show: Boolean = true
    39    @State het: number = 216
    40    @State wid: number = 384
    41    @State opacityValue: number = 0.4
    42    @State parmDynamicRangeModeStr: string[] = [
    43      'dynamicRangeMode = HIGH','dynamicRangeMode = CONSTRAINT','dynamicRangeMode = STANDARD'
    44    ]
    45    @State parmImageQualityStr: string[] = [
    46      'ResolutionQuality.LOW','ResolutionQuality.MEDIUM','ResolutionQuality.HIGH'
    47    ]
```

### After (local context)
```
    35  @Entry
    36  @Component
    37  struct DarkModeTest {
    38    modifier: ImageModifier = new ImageModifier()
    39    @State show: Boolean = true
    40    // 直接使用一般变量即可
    41    het: number = 216
    42    // 直接使用一般变量即可
    43    wid: number = 384
    44    opacityValue: number = 0.4
    45    @State parmDynamicRangeModeStr: string[] = [
    46      'dynamicRangeMode = HIGH','dynamicRangeMode = CONSTRAINT','dynamicRangeMode = STANDARD'
    47    ]
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/ImageHdrExamples/ImageHdrExample006.ets
+++ after/entry/src/main/ets/pages/ImageHdrExamples/ImageHdrExample006.ets
@@ -1,3 +1,4 @@
+  applyNormalAttribute(instance: ImageAttribute): void {
     instance.dynamicRangeMode(this.value)
     instance.borderRadius(this.bo)
     instance.enhancedImageQuality(this.qulity)
@@ -6,20 +7,19 @@
 @Entry
 @Component
 struct DarkModeTest {
-  @State modifier: ImageModifier = new ImageModifier()
+  modifier: ImageModifier = new ImageModifier()
   @State show: Boolean = true
-  @State het: number = 216
-  @State wid: number = 384
-  @State opacityValue: number = 0.4
+  // 直接使用一般变量即可
+  het: number = 216
+  // 直接使用一般变量即可
+  wid: number = 384
+  opacityValue: number = 0.4
   @State parmDynamicRangeModeStr: string[] = [
     'dynamicRangeMode = HIGH','dynamicRangeMode = CONSTRAINT','dynamicRangeMode = STANDARD'
   ]
   @State parmImageQualityStr: string[] = [
     'ResolutionQuality.LOW','ResolutionQuality.MEDIUM','ResolutionQuality.HIGH'
   ]
-  @State parmRenderModeStr: string[] = [
+  parmRenderModeStr: string[] = [
     'ImageRenderMode.Origin','ImageRenderMode.Template'
   ]
-  @State parmDynamicRangeMode: DynamicRangeMode[] = [
-    DynamicRangeMode.HIGH, DynamicRangeMode.CONSTRAINT, DynamicRangeMode.STANDARD
-  ]
```

## 7f3e3f9c30c5
- project: `Info`
- round_fixed: `1` (early)
- rule: `performance/foreach-args-check` (warn)
- file: `entry/src/main/ets/pages/loadingProgressStatic/LoadingProgressParallelization.ets`:321:11
- patch: shape=`modify`, changed_lines=182

### Before (local context)
```
   315                .enableLoading(true)
   316                .layoutWeight(1)
   317            })
   318          }
   319  
   320          ParallelizeUI({ enable: true }) {
   321            ForEach(forEachItems100, (item: Int, index: Double) => {
   322              LoadingProgress()
   323                .color('#fff1091b')
   324                .size({ width: '100%' })
   325                .enableLoading(true)
   326                .layoutWeight(1)
   327            })
```

### After (local context)
```
   315    values: number[] = [];
   316  
   317    constructor(values: number[]) {
   318      this.values = values
   319    }
   320  
   321    applyContent(): WrappedBuilder<LoadingProgressBuilder> {
   322      return wrapBuilder(buildLoadingProgress);
   323    }
   324  }
   325  
   326  @Builder
   327  function buildLoadingProgress(config: LoadingProgressConfiguration): void {
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/loadingProgressStatic/LoadingProgressParallelization.ets
+++ after/entry/src/main/ets/pages/loadingProgressStatic/LoadingProgressParallelization.ets
@@ -1,25 +1,25 @@
+      }
+    }
+  }
+}
 
-        ParallelizeUI({ enable: true }) {
-          ForEach(forEachItems100, (item: Int, index: Double) => {
-            LoadingProgress()
-              .color('#fff1091b')
-              .size({ width: '100%' })
-              .enableLoading(true)
-              .layoutWeight(1)
-          })
-        }
+class MyLoadingProgressStyle implements ContentModifier<LoadingProgressConfiguration> {
+  values: number[] = [];
 
-        ParallelizeUI({ enable: true }) {
-          ForEach(forEachItems100, (item: Int, index: Double) => {
-            LoadingProgress()
-              .color('#fff1091b')
-              .size({ width: '100%' })
-              .enableLoading(true)
-              .layoutWeight(1)
-          })
-        }
+  constructor(values: number[]) {
+    this.values = values
+  }
 
-        ParallelizeUI({ enable: true }) {
-          ForEach(forEachItems100, (item: Int, index: Double) => {
-            LoadingProgress()
-              .color('#fff1091b')
+  applyContent(): WrappedBuilder<LoadingProgressBuilder> {
+    return wrapBuilder(buildLoadingProgress);
+  }
+}
+
+@Builder
+function buildLoadingProgress(config: LoadingProgressConfiguration): void {
+  Column() {
+    Column() {
+    }
+
+    Column() {
+      Line().width("100%").backgroundColor("#ff373737")
```

## 27dca6b5f2a6
- project: `TextComponentTest`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-redundant-state-var` (suggestion)
- file: `entry/src/main/ets/pages/20-dev/Search.ets`:46:3
- patch: shape=`modify`, changed_lines=27

### Before (local context)
```
    40    @State letterSpacing: (number | string | Resource)[] = [-2, 0, 3, '5px', '10%']
    41    @State letterSpacingIndex: number = 4
    42    @State lineHeight: (number | string | Resource)[] = [-5, 0, 10, 20, '40vp', '30%']
    43    @State lineHeightIndex: number = 0
    44    @State cancelButton: CancelButtonStyle[] = [CancelButtonStyle.INPUT, CancelButtonStyle.CONSTANT, CancelButtonStyle.INVISIBLE]
    45    @State cancelButtonIndex: number = 0
    46    @State heightTest: number = 80
    47    @State widthTest: number = 250
    48    @State paddingTest: number = 0
    49    @State sizeAll: (number | string | Resource | undefined)[] = [-1, 0, 20, 'hsp', undefined, '20px', '10abc', $r('app.string.number_20px'), $r('app.string.number_10abc')]
    50    @State maxFontSize: number = 2
    51    @State minFontSize: number = 2
    52    @State heightAdaptivePolicy: TextHeightAdaptivePolicy[] = [TextHeightAdaptivePolicy.MAX_LINES_FIRST, TextHeightAdaptivePolicy.MIN_FONT_SIZE_FIRST, TextHeightAdaptivePolicy.LAYOUT_CONSTRAINT_FIRST]
```

### After (local context)
```
    40    maxLength: number = 6
    41    letterSpacing: (number | string | Resource)[] = [-2, 0, 3, '5px', '10%']
    42    letterSpacingIndex: number = 4
    43    lineHeight: (number | string | Resource)[] = [-5, 0, 10, 20, '40vp', '30%']
    44    lineHeightIndex: number = 0
    45    cancelButton: CancelButtonStyle[] = [CancelButtonStyle.INPUT, CancelButtonStyle.CONSTANT, CancelButtonStyle.INVISIBLE]
    46    cancelButtonIndex: number = 0
    47    heightTest: number = 80
    48    widthTest: number = 250
    49    paddingTest: number = 0
    50    sizeAll: (number | string | Resource | undefined)[] = [-1, 0, 20, 'hsp', undefined, '20px', '10abc', $r('app.string.number_20px'), $r('app.string.number_10abc')]
    51    maxFontSize: number = 2
    52    minFontSize: number = 2
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/20-dev/Search.ets
+++ after/entry/src/main/ets/pages/20-dev/Search.ets
@@ -1,25 +1,25 @@
-  @State textAlign: TextAlign[] = [TextAlign.Start, TextAlign.Center, TextAlign.End]
-  @State textAlignStr: string[] = ['Start', 'Center', 'End']
-  @State textAlignIndex: number = 0
-  @State fontSize:number = 16
-  @State showUnderline: boolean = false
-  @State maxLength: number = 6
-  @State letterSpacing: (number | string | Resource)[] = [-2, 0, 3, '5px', '10%']
-  @State letterSpacingIndex: number = 4
-  @State lineHeight: (number | string | Resource)[] = [-5, 0, 10, 20, '40vp', '30%']
-  @State lineHeightIndex: number = 0
-  @State cancelButton: CancelButtonStyle[] = [CancelButtonStyle.INPUT, CancelButtonStyle.CONSTANT, CancelButtonStyle.INVISIBLE]
-  @State cancelButtonIndex: number = 0
-  @State heightTest: number = 80
-  @State widthTest: number = 250
-  @State paddingTest: number = 0
-  @State sizeAll: (number | string | Resource | undefined)[] = [-1, 0, 20, 'hsp', undefined, '20px', '10abc', $r('app.string.number_20px'), $r('app.string.number_10abc')]
-  @State maxFontSize: number = 2
-  @State minFontSize: number = 2
-  @State heightAdaptivePolicy: TextHeightAdaptivePolicy[] = [TextHeightAdaptivePolicy.MAX_LINES_FIRST, TextHeightAdaptivePolicy.MIN_FONT_SIZE_FIRST, TextHeightAdaptivePolicy.LAYOUT_CONSTRAINT_FIRST]
-  @State heightAdaptiveIndex: number = 0
+  style: TextContentStyle = TextContentStyle.DEFAULT
+  textAlign: TextAlign[] = [TextAlign.Start, TextAlign.Center, TextAlign.End]
+  textAlignStr: string[] = ['Start', 'Center', 'End']
+  textAlignIndex: number = 0
+  fontSize:number = 16
+  showUnderline: boolean = false
+  maxLength: number = 6
+  letterSpacing: (number | string | Resource)[] = [-2, 0, 3, '5px', '10%']
+  letterSpacingIndex: number = 4
+  lineHeight: (number | string | Resource)[] = [-5, 0, 10, 20, '40vp', '30%']
+  lineHeightIndex: number = 0
+  cancelButton: CancelButtonStyle[] = [CancelButtonStyle.INPUT, CancelButtonStyle.CONSTANT, CancelButtonStyle.INVISIBLE]
+  cancelButtonIndex: number = 0
+  heightTest: number = 80
+  widthTest: number = 250
+  paddingTest: number = 0
+  sizeAll: (number | string | Resource | undefined)[] = [-1, 0, 20, 'hsp', undefined, '20px', '10abc', $r('app.string.number_20px'), $r('app.string.number_10abc')]
+  maxFontSize: number = 2
+  minFontSize: number = 2
+  heightAdaptivePolicy: TextHeightAdaptivePolicy[] = [TextHeightAdaptivePolicy.MAX_LINES_FIRST, TextHeightAdaptivePolicy.MIN_FONT_SIZE_FIRST, TextHeightAdaptivePolicy.LAYOUT_CONSTRAINT_FIRST]
+  heightAdaptiveIndex: number = 0
   @State selectionStart: number = 0
   @State selectionEnd: number = 0
 
   build() {
-    Column() {
```

## 86ccfc1468f9
- project: `ace_ets_module_imageText_api12`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-redundant-state-var` (suggestion)
- file: `entry/src/main/ets/MainAbility/pages/RichEditor/RichEditorStyledStringTest2.ets`:22:3
- patch: shape=`modify`, changed_lines=3

### Before (local context)
```
    16  import { LengthMetrics } from '@ohos.arkui.node'
    17  
    18  @Entry
    19  @Component
    20  struct RichEditorStyledStringTest2 {
    21    stringLength: number = 0;
    22    @State selection: string = "";
    23    @State content: string = "";
    24    @State range: string = "";
    25    @State replaceString: string = "";
    26    @State rangeBefore: string = "";
    27    @State rangeAfter: string = "";
    28    richEditorStyledString: MutableStyledString = new MutableStyledString("");
```

### After (local context)
```
    16  
    17  import { LengthMetrics } from '@ohos.arkui.node'
    18  
    19  @Entry
    20  @Component
    21  struct RichEditorStyledStringTest2 {
    22    stringLength: number = 0;
    23    @State range: string = "";
    24    @State replaceString: string = "";
    25    @State rangeBefore: string = "";
    26    @State rangeAfter: string = "";
    27    richEditorStyledString: MutableStyledString = new MutableStyledString("");
    28    textStyle: TextStyle = new TextStyle({
```

### Local diff
```diff
--- before/entry/src/main/ets/MainAbility/pages/RichEditor/RichEditorStyledStringTest2.ets
+++ after/entry/src/main/ets/MainAbility/pages/RichEditor/RichEditorStyledStringTest2.ets
@@ -1,3 +1,4 @@
+ * Unless required by applicable law or agreed to in writing, software
  * distributed under the License is distributed on an "AS IS" BASIS,
  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  * See the License for the specific language governing permissions and
@@ -10,8 +11,6 @@
 @Component
 struct RichEditorStyledStringTest2 {
   stringLength: number = 0;
-  @State selection: string = "";
-  @State content: string = "";
   @State range: string = "";
   @State replaceString: string = "";
   @State rangeBefore: string = "";
@@ -23,3 +22,4 @@
     fontColor: Color.Green,
     fontSize: LengthMetrics.vp(30),
     fontStyle: FontStyle.Normal
+  })
```

## b984bf24c067
- project: `ace_ets_module_imageText_api12`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-redundant-state-var` (suggestion)
- file: `entry/src/main/ets/MainAbility/pages/RichEditorStringSpan/RichEditorStringSpan003.ets`:27:3
- patch: shape=`delete`, changed_lines=5

### Before (local context)
```
    21  struct RichEditorStringSpan003 {
    22    stringLength: number = 0;
    23    @State selection: string = '';
    24    @State content: string = '';
    25    @State range: string = '';
    26    @State replaceString: string = '';
    27    @State rangeBefore: string = '';
    28    @State rangeAfter: string = '';
    29    richEditorStyledString: MutableStyledString = new MutableStyledString('');
    30    textStyle: TextStyle = new TextStyle({
    31      fontWeight: FontWeight.Lighter,
    32      fontFamily: Utils.FONT_FAMILY,
    33      fontColor: Color.Green,
```

### After (local context)
```
    21  struct RichEditorStringSpan003 {
    22    stringLength: number = 0;
    23    @State content: string = '';
    24    richEditorStyledString: MutableStyledString = new MutableStyledString('');
    25    textStyle: TextStyle = new TextStyle({
    26      fontWeight: FontWeight.Lighter,
    27      fontFamily: Utils.FONT_FAMILY,
    28      fontColor: Color.Green,
    29      fontSize: LengthMetrics.vp(30),
    30      fontStyle: FontStyle.Normal
    31    })
    32    fontStyle1: TextStyle = new TextStyle({ fontColor: Color.Blue });
    33    fontStyle2: TextStyle = new TextStyle({
```

### Local diff
```diff
--- before/entry/src/main/ets/MainAbility/pages/RichEditorStringSpan/RichEditorStringSpan003.ets
+++ after/entry/src/main/ets/MainAbility/pages/RichEditorStringSpan/RichEditorStringSpan003.ets
@@ -6,12 +6,7 @@
 @Component
 struct RichEditorStringSpan003 {
   stringLength: number = 0;
-  @State selection: string = '';
   @State content: string = '';
-  @State range: string = '';
-  @State replaceString: string = '';
-  @State rangeBefore: string = '';
-  @State rangeAfter: string = '';
   richEditorStyledString: MutableStyledString = new MutableStyledString('');
   textStyle: TextStyle = new TextStyle({
     fontWeight: FontWeight.Lighter,
@@ -23,3 +18,8 @@
   fontStyle1: TextStyle = new TextStyle({ fontColor: Color.Blue });
   fontStyle2: TextStyle = new TextStyle({
     fontWeight: FontWeight.Bolder,
+    fontFamily: 'Arial',
+    fontColor: Color.Orange,
+    fontSize: LengthMetrics.vp(50),
+    fontStyle: FontStyle.Italic
+  })
```

## 67bea9a7bf49
- project: `ace_ets_module_nowear_waterflow`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-unchanged-state-var` (suggestion)
- file: `entry/src/main/ets/MainAbility/pages/onScrollIndex/onScrollIndexFlow2.ets`:19:3
- patch: shape=`modify`, changed_lines=16

### Before (local context)
```
    13   * limitations under the License.
    14   */
    15  import { WaterFlowDataSource } from '../WaterFlowDataSource'
    16  @Entry
    17  @Component
    18  struct onScrollIndexFlow2 {
    19    @State minSize: number = 80
    20    @State maxSize: number = 180
    21    @State temp: number = 0;
    22    scroller: Scroller = new Scroller()
    23    @State colors: number[] = [0xFFC0CB, 0xDA70D6, 0x6B8E23, 0x6A5ACD, 0x00FFFF, 0x00FF7F]
    24    dataSource: WaterFlowDataSource = new WaterFlowDataSource()
    25    @State layoutMode1: WaterFlowLayoutMode = WaterFlowLayoutMode.ALWAYS_TOP_DOWN
```

### After (local context)
```
    13   * limitations under the License.
    14   */
    15  import { WaterFlowDataSource } from '../WaterFlowDataSource'
    16  @Reusable
    17  @Component
    18  struct FlowItemComponent {
    19    item: number = 0
    20  
    21    build() {
    22      Text('N' + this.item).fontSize(12).height('16')
    23    }
    24  }
    25  @Entry
```

### Local diff
```diff
--- before/entry/src/main/ets/MainAbility/pages/onScrollIndex/onScrollIndexFlow2.ets
+++ after/entry/src/main/ets/MainAbility/pages/onScrollIndex/onScrollIndexFlow2.ets
@@ -7,19 +7,19 @@
  * limitations under the License.
  */
 import { WaterFlowDataSource } from '../WaterFlowDataSource'
+@Reusable
+@Component
+struct FlowItemComponent {
+  item: number = 0
+
+  build() {
+    Text('N' + this.item).fontSize(12).height('16')
+  }
+}
 @Entry
 @Component
 struct onScrollIndexFlow2 {
-  @State minSize: number = 80
-  @State maxSize: number = 180
+  minSize: number = 80
+  maxSize: number = 180
   @State temp: number = 0;
   scroller: Scroller = new Scroller()
-  @State colors: number[] = [0xFFC0CB, 0xDA70D6, 0x6B8E23, 0x6A5ACD, 0x00FFFF, 0x00FF7F]
-  dataSource: WaterFlowDataSource = new WaterFlowDataSource()
-  @State layoutMode1: WaterFlowLayoutMode = WaterFlowLayoutMode.ALWAYS_TOP_DOWN
-  private itemWidthArray: number[] = []
-  private itemHeightArray: number[] = []
-  @State onScrollIndex:string = ''
-
-  // 计算FlowItem宽/高
-  getSize() {
```

## f66cbe28e453
- project: `ace_ets_module_router1`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-unchanged-state-var` (suggestion)
- file: `entry/src/main/ets/MainAbility/pages/Router2/stageRouter.ets`:22:3
- patch: shape=`delete`, changed_lines=2

### Before (local context)
```
    16  @Entry
    17  @Component
    18  struct stageRouter {
    19    private TAG: string = '[AnimatorTest]';
    20    private backAnimator: ESObject = undefined;
    21    private flag: boolean = false;
    22    @State wid: number = 100;
    23    @State hei: number = 100;
    24  
    25    build() {
    26      Flex({ direction: FlexDirection.Column, alignItems: ItemAlign.Center, justifyContent: FlexAlign.Center }) {
    27  
    28        Button('router-pushUrl')
```

### After (local context)
```
    16  @Entry
    17  @Component
    18  struct stageRouter {
    19    private TAG: string = '[AnimatorTest]';
    20    private backAnimator: ESObject = undefined;
    21    private flag: boolean = false;
    22  
    23    build() {
    24      Flex({ direction: FlexDirection.Column, alignItems: ItemAlign.Center, justifyContent: FlexAlign.Center }) {
    25  
    26        Button('router-pushUrl')
    27          .width(100)
    28          .height(70)
```

### Local diff
```diff
--- before/entry/src/main/ets/MainAbility/pages/Router2/stageRouter.ets
+++ after/entry/src/main/ets/MainAbility/pages/Router2/stageRouter.ets
@@ -10,8 +10,6 @@
   private TAG: string = '[AnimatorTest]';
   private backAnimator: ESObject = undefined;
   private flag: boolean = false;
-  @State wid: number = 100;
-  @State hei: number = 100;
 
   build() {
     Flex({ direction: FlexDirection.Column, alignItems: ItemAlign.Center, justifyContent: FlexAlign.Center }) {
@@ -23,3 +21,5 @@
         .fontColor(Color.Black)
         .onClick(() => {
           globalThis.uiContent.getRouter().pushUrl({
+            url: 'pages/Animator',
+            params: {
```

## 296d09915e1a
- project: `ace_ets_module_router1`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-unchanged-state-var` (suggestion)
- file: `entry/src/main/ets/MainAbility/pages/router/getLength/getLength1.ets`:21:3
- patch: shape=`modify`, changed_lines=2

### Before (local context)
```
    15  import router from '@ohos.router'
    16  import { BusinessError } from '@ohos.base'
    17  
    18  @Entry
    19  @Component
    20  struct getLength1 {
    21    @State str:string = 'getLength1'
    22    @State sizeValue:string = ''
    23  
    24    build() {
    25      Column({space:5}) {
    26  
    27        Text(this.str).id('getLength1_text')
```

### After (local context)
```
    15  import router from '@ohos.router'
    16  import { BusinessError } from '@ohos.base'
    17  
    18  @Entry
    19  @Component
    20  struct getLength1 {
    21     // str 未发生变化，改为普通变量
    22     str:string = 'getLength1'
    23    @State sizeValue:string = ''
    24  
    25    build() {
    26      Column({space:5}) {
    27  
```

### Local diff
```diff
--- before/entry/src/main/ets/MainAbility/pages/router/getLength/getLength1.ets
+++ after/entry/src/main/ets/MainAbility/pages/router/getLength/getLength1.ets
@@ -10,7 +10,8 @@
 @Entry
 @Component
 struct getLength1 {
-  @State str:string = 'getLength1'
+   // str 未发生变化，改为普通变量
+   str:string = 'getLength1'
   @State sizeValue:string = ''
 
   build() {
@@ -22,4 +23,3 @@
       Button('getLength')
         .id('getLength1_get')
         .onClick(()=>{
-          this.sizeValue = router.getLength()
```

## 101fb873d819
- project: `ace_ets_module_router1`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-redundant-state-var` (suggestion)
- file: `entry/src/main/ets/MainAbility/pages/router/router_getStateByUrl2.ets`:20:3
- patch: shape=`delete`, changed_lines=1

### Before (local context)
```
    14   */
    15  import router from '@ohos.router'
    16  
    17  @Entry
    18  @Component
    19  struct router_getStateByUrl2 {
    20    @State str: string = ''
    21  
    22    build() {
    23      Column({ space: 5 }) {
    24  
    25        Button('pushUrl')
    26          .id('router_getStateByUrl2_btn')
```

### After (local context)
```
    14   */
    15  import router from '@ohos.router'
    16  
    17  @Entry
    18  @Component
    19  struct router_getStateByUrl2 {
    20  
    21    build() {
    22      Column({ space: 5 }) {
    23  
    24        Button('pushUrl')
    25          .id('router_getStateByUrl2_btn')
    26          .onClick(() => {
```

### Local diff
```diff
--- before/entry/src/main/ets/MainAbility/pages/router/router_getStateByUrl2.ets
+++ after/entry/src/main/ets/MainAbility/pages/router/router_getStateByUrl2.ets
@@ -10,7 +10,6 @@
 @Entry
 @Component
 struct router_getStateByUrl2 {
-  @State str: string = ''
 
   build() {
     Column({ space: 5 }) {
@@ -23,3 +22,4 @@
 
     }.width('100%').height('100%')
 
+  }
```

## 26d93f0a8c05
- project: `acts_validator`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-unchanged-state-var` (suggestion)
- file: `entry/src/main/ets/pages/ArkUI/dialogModeTest.ets`:52:3
- patch: shape=`modify`, changed_lines=7

### Before (local context)
```
    46  }
    47  
    48  @Entry
    49  @Component
    50  struct dialogModeTest {
    51    @State name: string = 'dialogModeTest';
    52    @State stepTips: string =
    53      '操作步骤:\n' +
    54        '1、分别点击上方三个按钮切换immersiveMode的值，再分别点击下方橙色按钮弹出弹窗，并观察各类弹窗的蒙层效果\n' +
    55        '预期结果：\n' +
    56        '1、非自由子窗口中，当immersiveMode的值为EXTEND(1)时弹窗蒙层可扩展至覆盖状态栏和导航条，其余情况避开状态栏和导航条\n' +
    57        '2、自由窗口设备（PC端）直接按照失败处理走豁免'
    58    @State isVue: boolean = false;
```

### After (local context)
```
    46    }
    47  }
    48  
    49  @Entry
    50  @Component
    51  struct dialogModeTest {
    52    // 直接使用一般变量即可
    53    name: string = 'dialogModeTest';
    54    // 直接使用一般变量即可
    55    stepTips: string =
    56      '操作步骤:\n' +
    57        '1、分别点击上方三个按钮切换immersiveMode的值，再分别点击下方橙色按钮弹出弹窗，并观察各类弹窗的蒙层效果\n' +
    58        '预期结果：\n' +
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/ArkUI/dialogModeTest.ets
+++ after/entry/src/main/ets/pages/ArkUI/dialogModeTest.ets
@@ -1,3 +1,4 @@
+      Button('点我关闭弹窗')
         .onClick(() => {
           dialogController1?.close();
         })
@@ -9,17 +10,16 @@
 @Entry
 @Component
 struct dialogModeTest {
-  @State name: string = 'dialogModeTest';
-  @State stepTips: string =
+  // 直接使用一般变量即可
+  name: string = 'dialogModeTest';
+  // 直接使用一般变量即可
+  stepTips: string =
     '操作步骤:\n' +
       '1、分别点击上方三个按钮切换immersiveMode的值，再分别点击下方橙色按钮弹出弹窗，并观察各类弹窗的蒙层效果\n' +
       '预期结果：\n' +
       '1、非自由子窗口中，当immersiveMode的值为EXTEND(1)时弹窗蒙层可扩展至覆盖状态栏和导航条，其余情况避开状态栏和导航条\n' +
       '2、自由窗口设备（PC端）直接按照失败处理走豁免'
   @State isVue: boolean = false;
-  @State intervalNum: number = 0;
+  intervalNum: number = 0;
   @State immersiveMode: ImmersiveMode | undefined = 0
 
-  @Builder
-  PassBtn(text: Resource, isFullScreen: boolean) {
-    if (this.isVue == false) {
```

## 98980312a33d
- project: `applications_permission_manager`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-unchanged-state-var` (suggestion)
- file: `permissionmanager/src/main/ets/common/components/alphabeticalIndex.ets`:25:3
- patch: shape=`add`, changed_lines=1

### Before (local context)
```
    19  import { AppInfo, ApplicationObj } from '../model/typedef';
    20  
    21  @Component
    22  export struct alphabetIndexerComponent {
    23    @Link applicationItem: Array<AppInfo | ApplicationObj>; // application info array
    24    @Link index: number; // alphabetical index
    25    @State usingPopup: boolean = false;
    26  
    27    aboutToAppear() {
    28      const this_ = this;
    29      setTimeout(() => {
    30        this_.usingPopup = true;
    31      }, 1000)
```

### After (local context)
```
    19  import { GlobalContext } from '../utils/globalContext';
    20  import { AppInfo, ApplicationObj } from '../model/typedef';
    21  
    22  @Component
    23  export struct alphabetIndexerComponent {
    24    @Link applicationItem: Array<AppInfo | ApplicationObj>; // application info array
    25    @Link index: number; // alphabetical index
    26    @State usingPopup: boolean = false;
    27  
    28    aboutToAppear() {
    29      const this_ = this;
    30      setTimeout(() => {
    31        this_.usingPopup = true;
```

### Local diff
```diff
--- before/permissionmanager/src/main/ets/common/components/alphabeticalIndex.ets
+++ after/permissionmanager/src/main/ets/common/components/alphabeticalIndex.ets
@@ -1,3 +1,4 @@
+ * See the License for the specific language governing permissions and
  * limitations under the License.
  */
 
@@ -22,4 +23,3 @@
   build() {
     AlphabetIndexer({ arrayValue: indexValue, selected: this.index })
       .color($r('sys.color.font_secondary'))
-      .selectedColor($r('sys.color.font_emphasize')) // selected color
```

## 2f10087cda56
- project: `asn1_ber`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-use-local-var-to-replace-state-var` (warn)
- file: `entry/src/main/ets/pages/Index.ets`:665:5
- patch: shape=`modify`, changed_lines=382

### Before (local context)
```
   659        0x01, 0xff, 0x01, 0x01, 0xff
   660      ])
   661      writer.writeBuffer(expected.subarray(2, expected.length), 0x04)
   662      let buffer1: buffer.Buffer = writer.buffer;
   663  
   664      this.result += 'writeBuffer\r\n'
   665      this.result += '-->  write a value: ' + (buffer1.length == 26) + '\r\n'
   666      this.result += '-->  write a value: ' + (buffer1[0] == 0x04) + '\r\n'
   667      this.result += '-->  write a value: ' + (buffer1[1] == 11) + '\r\n'
   668      this.result += '-->  write a value: ' + (buffer1.subarray(2, 13).toString("utf8") == "hello world") + '\r\n'
   669      this.result += '-->  write a value: ' + (buffer1[13] == 0) + '\r\n'
   670      this.result += '-->  write a value: ' + (buffer1[14] == 1) + '\r\n'
   671      for (let i = 13, j = 0; i < buffer1.length && j < expected.length; i++, j++) {
```

### After (local context)
```
   659      tempResult += 'writeBoolean\r\n'
   660      tempResult += '-->  write a true and false value: ' + (buffer.length == 6) + '\r\n'
   661      tempResult += '-->  write a true and false value: ' + (buffer[0] == 0x01) + '\r\n'
   662      tempResult += '-->  write a true and false value: ' + (buffer[1] == 0x01) + '\r\n'
   663      tempResult += '-->  write a true and false value: ' + (buffer[2] == 0xff) + '\r\n'
   664      tempResult += '-->  write a true and false value: ' + (buffer[3] == 0x01) + '\r\n'
   665      tempResult += '-->  write a true and false value: ' + (buffer[4] == 0x01) + '\r\n'
   666      tempResult += '-->  write a true and false value: ' + (buffer[5] == 0x00) + '\r\n\r\n'
   667      this.result = tempResult;
   668    }
   669  
   670    private writeString() {
   671      let writer = new BerWriter()
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/Index.ets
+++ after/entry/src/main/ets/pages/Index.ets
@@ -1,25 +1,25 @@
+  private writeBoolean() {
+    let writer = new BerWriter()
+    writer.writeBoolean(true)
+    writer.writeBoolean(false)
+    let buffer: buffer.Buffer = writer.buffer
+    let tempResult = this.result;
+    tempResult += 'writeBoolean\r\n'
+    tempResult += '-->  write a true and false value: ' + (buffer.length == 6) + '\r\n'
+    tempResult += '-->  write a true and false value: ' + (buffer[0] == 0x01) + '\r\n'
+    tempResult += '-->  write a true and false value: ' + (buffer[1] == 0x01) + '\r\n'
+    tempResult += '-->  write a true and false value: ' + (buffer[2] == 0xff) + '\r\n'
+    tempResult += '-->  write a true and false value: ' + (buffer[3] == 0x01) + '\r\n'
+    tempResult += '-->  write a true and false value: ' + (buffer[4] == 0x01) + '\r\n'
+    tempResult += '-->  write a true and false value: ' + (buffer[5] == 0x00) + '\r\n\r\n'
+    this.result = tempResult;
+  }
 
-  private writeBuffer() {
+  private writeString() {
     let writer = new BerWriter()
     writer.writeString("hello world")
-    let expected: buffer.Buffer = buffer.from([
-      0x04, 0x0b, 0x30, 0x09, 0x02, 0x01, 0x0f, 0x01,
-      0x01, 0xff, 0x01, 0x01, 0xff
-    ])
-    writer.writeBuffer(expected.subarray(2, expected.length), 0x04)
-    let buffer1: buffer.Buffer = writer.buffer;
-
-    this.result += 'writeBuffer\r\n'
-    this.result += '-->  write a value: ' + (buffer1.length == 26) + '\r\n'
-    this.result += '-->  write a value: ' + (buffer1[0] == 0x04) + '\r\n'
-    this.result += '-->  write a value: ' + (buffer1[1] == 11) + '\r\n'
-    this.result += '-->  write a value: ' + (buffer1.subarray(2, 13).toString("utf8") == "hello world") + '\r\n'
-    this.result += '-->  write a value: ' + (buffer1[13] == 0) + '\r\n'
-    this.result += '-->  write a value: ' + (buffer1[14] == 1) + '\r\n'
-    for (let i = 13, j = 0; i < buffer1.length && j < expected.length; i++, j++) {
-      this.result += '-->  write a value: ' + (buffer1[i] == expected[j]) + '\r\n'
-    }
-  }
-
-  private writeStringArray() {
-    let writer = new BerWriter()
+    let buffer: buffer.Buffer = writer.buffer
+    let tempResult = this.result;
+    tempResult += 'writeString\r\n'
+    tempResult += '-->  write a value: ' + (buffer.length == 13) + '\r\n'
+    tempResult += '-->  write a value: ' + (buffer[0] == 0x04) + '\r\n'
```

## 6b2b496008c7
- project: `bluetoothtest`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-unchanged-state-var` (suggestion)
- file: `entry/src/main/ets/Component/bleFilterTable.ets`:27:3
- patch: shape=`modify`, changed_lines=61

### Before (local context)
```
    21   */
    22  
    23  @Component
    24  export struct BleFilterTable {
    25    private testItem!: TestData
    26    @State apiItems: TestApi[] = initBRApiData()
    27    @State changeIndex: number = - 1
    28    // input ble scan parameters:
    29    /*ScanFilter*/
    30    @State cbxBleScanFilter: boolean = false;
    31    @State txtScanFilter_deviceId: string = '6c:96:d7:3d:87:6f'; //6c:96:d7:3d:87:6f
    32    @State txtScanFilter_name: string = "dudu-tiger";
    33    @State txtScanFilter_serviceUuid: string = "0000180A-0000-1000-8000-00805f9b34fb";
```

### After (local context)
```
    21   * BleFilterTable Of Bluetooth test
    22   */
    23  
    24  @Component
    25  export struct BleFilterTable {
    26    private testItem!: TestData
    27    apiItems: TestApi[] = initBRApiData()
    28    // changeIndex 未发生变化，改为普通变量
    29    changeIndex: number = - 1
    30    // input ble scan parameters:
    31    /*ScanFilter*/
    32    @State cbxBleScanFilter: boolean = false;
    33    @State txtScanFilter_deviceId: string = '6c:96:d7:3d:87:6f'; //6c:96:d7:3d:87:6f
```

### Local diff
```diff
--- before/entry/src/main/ets/Component/bleFilterTable.ets
+++ after/entry/src/main/ets/Component/bleFilterTable.ets
@@ -1,3 +1,4 @@
+ */
 
 import { TestData , TestApi } from '../MainAbility/model/testData'
 import { initBRApiData } from '../MainAbility/model/testDataModels'
@@ -9,8 +10,9 @@
 @Component
 export struct BleFilterTable {
   private testItem!: TestData
-  @State apiItems: TestApi[] = initBRApiData()
-  @State changeIndex: number = - 1
+  apiItems: TestApi[] = initBRApiData()
+  // changeIndex 未发生变化，改为普通变量
+  changeIndex: number = - 1
   // input ble scan parameters:
   /*ScanFilter*/
   @State cbxBleScanFilter: boolean = false;
@@ -21,5 +23,3 @@
   @State cbxBleScanOptions: boolean = false;
   @State txtScanOptions_interval: string = "0";
 
-  getCurrentState(index: number) {
-    return this.apiItems[ index ].result
```

## a80cfdcd5d6d
- project: `bluetoothtest`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-redundant-state-var` (suggestion)
- file: `entry/src/main/ets/pages/manualApiTestPage.ets`:25:3
- patch: shape=`modify`, changed_lines=2

### Before (local context)
```
    19  /**
    20   * Manual API Test Page Of Bluetooth test
    21   */
    22  @Entry
    23  @Component
    24  struct ManualApiTestPage {
    25    @State message: string = 'ManualApiTest'
    26  
    27    build() {
    28      Column() {
    29        GridContainer({
    30          columns : 12 ,
    31          sizeType : SizeType.Auto ,
```

### After (local context)
```
    19  import { SubEntryComponent } from '../Component/subEntryComponent';
    20  /**
    21   * Manual API Test Page Of Bluetooth test
    22   */
    23  @Entry
    24  @Component
    25  struct ManualApiTestPage {
    26  
    27    build() {
    28      Column() {
    29        GridContainer({
    30          columns : 12 ,
    31          sizeType : SizeType.Auto ,
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/manualApiTestPage.ets
+++ after/entry/src/main/ets/pages/manualApiTestPage.ets
@@ -1,3 +1,4 @@
+ * See the License for the specific language governing permissions and
  * limitations under the License.
  */
 
@@ -10,7 +11,6 @@
 @Entry
 @Component
 struct ManualApiTestPage {
-  @State message: string = 'ManualApiTest'
 
   build() {
     Column() {
```

## 5bcbf5e4d906
- project: `shopping`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-redundant-state-var` (suggestion)
- file: `entry/src/main/ets/pages/collect/collected.ets`:29:3
- patch: shape=`modify`, changed_lines=110

### Before (local context)
```
    23  
    24  @Entry
    25  @Component
    26  struct CollectedPage {
    27    @State width1: number = 10
    28    @State ratio: number = 1
    29    @State opacity1: number = 1
    30    @State num: number = 1
    31    @State layoutId: number = 0
    32    @State collected: boolean = true
    33    // listener = mediaQuery.matchMediaSync('(orientation:landscape)')
    34  
    35    onPageShow() {
```

### After (local context)
```
    23  
    24  @Entry
    25  @Component
    26  struct CollectedPage {
    27    width1: number = 10
    28    @State ratio: number = 1
    29    opacity1: number = 1
    30    // 直接使用一般变量即可
    31    num: number = 1
    32    @State layoutId: number = 0
    33    // 直接使用一般变量即可
    34    collected: boolean = true
    35  
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/collect/collected.ets
+++ after/entry/src/main/ets/pages/collect/collected.ets
@@ -8,13 +8,14 @@
 @Entry
 @Component
 struct CollectedPage {
-  @State width1: number = 10
+  width1: number = 10
   @State ratio: number = 1
-  @State opacity1: number = 1
-  @State num: number = 1
+  opacity1: number = 1
+  // 直接使用一般变量即可
+  num: number = 1
   @State layoutId: number = 0
-  @State collected: boolean = true
-  // listener = mediaQuery.matchMediaSync('(orientation:landscape)')
+  // 直接使用一般变量即可
+  collected: boolean = true
 
   onPageShow() {
     const params = router.getParams() as Record<number, string>; // 获取传递过来的参数对象
@@ -22,4 +23,3 @@
     this.ratio = params['ratio']
   }
 
-  aboutToAppear() {
```

## 7419daa9264f
- project: `YFree_HarmonyOS`
- round_fixed: `1` (early)
- rule: `security/no-commented-code` (warn)
- file: `YFree/src/main/ets/network.ets`:38:1
- patch: shape=`add`, changed_lines=1

### Before (local context)
```
    32    jkid: string
    33    jkparams?: Record<string, Object | null> | null
    34    showProgress?: boolean
    35    showErrorToast?: boolean
    36  }
    37  
    38  /*
    39   * yPost(
    40        {
    41          jkid: "01A01",
    42          jkparams: {},
    43        },
    44        (isSuccess, results) => {
```

### After (local context)
```
    32  interface _YPostOption {
    33    jkid: string
    34    jkparams?: Record<string, Object | null> | null
    35    showProgress?: boolean
    36    showErrorToast?: boolean
    37  }
    38  
    39  /*
    40   * yPost(
    41        {
    42          jkid: "01A01",
    43          jkparams: {},
    44        },
```

### Local diff
```diff
--- before/YFree/src/main/ets/network.ets
+++ after/YFree/src/main/ets/network.ets
@@ -1,3 +1,4 @@
+class _YResponseModel {
   result: Object[] = []
   code: string = "y_code"
   message: string = ""
@@ -22,4 +23,3 @@
         }
       },
       UserModel,
-    )
```

## 311d335bf0e7
- project: `YFree_HarmonyOS`
- round_fixed: `1` (early)
- rule: `security/no-commented-code` (warn)
- file: `YFree/src/main/ets/storage.ets`:27:1
- patch: shape=`modify`, changed_lines=16

### Before (local context)
```
    21  // }
    22  //
    23  // export async function yRead(key: string): Promise<string | null> {
    24  //   try {
    25  //     return YApplication.yPreferences.getSync(key, null)?.toString()
    26  //   } catch (e) {
    27  //     return null
    28  //   }
    29  // }
    30  //
    31  // export async function yDelete(key: string) {
    32  //   YApplication.yPreferences.deleteSync(key)
    33  //   await YApplication.yPreferences.flush()
```

### After (local context)
```
    21  
    22  export async function ySave(key: string, value: string) {
    23    await YApplication.yKVStore.put(key, value)
    24  }
    25  
    26  export async function yRead(key: string): Promise<string | null> {
    27    return await yRunCompleter((completer) => {
    28      YApplication.yKVStore.get(key, (err, data) => {
    29        completer(data as string | null)
    30      })
    31    })
    32  }
    33  
```

### Local diff
```diff
--- before/YFree/src/main/ets/storage.ets
+++ after/YFree/src/main/ets/storage.ets
@@ -1,25 +1,25 @@
+*/
 
 import { YApplication, yRunCompleter } from '../../../index';
 
-// export async function ySave(key: string, value: string) {
-//   YApplication.yPreferences.putSync(key, value)
-//   await YApplication.yPreferences.flush()
-// }
 //
-// export async function yRead(key: string): Promise<string | null> {
-//   try {
-//     return YApplication.yPreferences.getSync(key, null)?.toString()
-//   } catch (e) {
-//     return null
-//   }
-// }
 //
-// export async function yDelete(key: string) {
-//   YApplication.yPreferences.deleteSync(key)
-//   await YApplication.yPreferences.flush()
-// }
 
 export async function ySave(key: string, value: string) {
   await YApplication.yKVStore.put(key, value)
 }
 
+export async function yRead(key: string): Promise<string | null> {
+  return await yRunCompleter((completer) => {
+    YApplication.yKVStore.get(key, (err, data) => {
+      completer(data as string | null)
+    })
+  })
+}
+
+export async function yDelete(key: string) {
+  return await yRunCompleter<Object | null>((completer) => {
+    YApplication.yKVStore.delete(key, (err) => {
+      completer(null)
+    })
+  })
```

## 8cd42ee946af
- project: `flutter_embedding`
- round_fixed: `1` (early)
- rule: `security/no-cycle` (warn)
- file: `flutter/src/main/ets/plugin/PlatformPlugin.ets`:12:1
- patch: shape=`modify`, changed_lines=2

### Before (local context)
```
     6  * Based on PlatformPlugin.java originally written by
     7  * Copyright (C) 2013 The Flutter Authors.
     8  *
     9  */
    10  import abilityAccessCtrl from '@ohos.abilityAccessCtrl';
    11  import { BusinessError } from '@kit.BasicServicesKit';
    12  import PlatformChannel, {
    13    AppSwitcherDescription,
    14    Brightness,
    15    ClipboardContentFormat,
    16    HapticFeedbackType,
    17    PlatformMessageHandler,
    18    SoundType,
```

### After (local context)
```
     6  *
     7  * Based on PlatformPlugin.java originally written by
     8  * Copyright (C) 2013 The Flutter Authors.
     9  *
    10  */
    11  import abilityAccessCtrl from '@ohos.abilityAccessCtrl';
    12  import { BusinessError } from '@kit.BasicServicesKit';
    13  import PlatformChannel, {
    14    AppSwitcherDescription,
    15    Brightness,
    16    ClipboardContentFormat,
    17    HapticFeedbackType,
    18    PlatformMessageHandler,
```

### Local diff
```diff
--- before/flutter/src/main/ets/plugin/PlatformPlugin.ets
+++ after/flutter/src/main/ets/plugin/PlatformPlugin.ets
@@ -1,3 +1,4 @@
+
 /*
 * Copyright (c) 2023 Hunan OpenValley Digital Industry Development Co., Ltd. All rights reserved.
 * Use of this source code is governed by a BSD-style license that can be
@@ -19,6 +20,5 @@
   SystemChromeStyle,
   SystemUiMode,
   SystemUiOverlay
-} from '../embedding/engine/systemchannels/PlatformChannel';
+} from '../embedding/engine/systemchannels/PlatformChannelNoCycle';
 import FlutterManager from '../embedding/ohos/FlutterManager';
-import pasteboard from '@ohos.pasteboard';
```

## 8de3d006ad87
- project: `ohos_mail_base`
- round_fixed: `1` (early)
- rule: `security/no-commented-code` (warn)
- file: `lib/src/main/ets/format/msg.ts`:478:11
- patch: shape=`modify`, changed_lines=27

### Before (local context)
```
   472            //   if (entry.name.startsWith("__substg1.0_")) {
   473            //     let pa: DirectoryItem = this.parseDirectory(entry);
   474            //     logger.trace("__nameid_1",
   475            //       pa.name,
   476            //       pa.value,
   477            //       pa.typeCode.toString(16),
   478            //       entry.startSectorLocation,
   479            //       entry.streamSize
   480            //     );
   481            //   }
   482            // }
   483          }
   484          break;
```

### After (local context)
```
   472      this._cfb.parse()
   473      const t = this.parseEntry(this._cfb.getRootEntry());
   474      return t;
   475    }
   476  
   477    parseEntry(entry: DirectoryEntry) {
   478      if (entry.objectType == DirectoryObjectType.Stream) {
   479        const o = this.parseDirectory(entry);
   480        return o;
   481      } else {
   482        const children = this._cfb.readStorage(entry).map(e => this.parseEntry(e));
   483        const o = {
   484          name: entry.name,
```

### Local diff
```diff
--- before/lib/src/main/ets/format/msg.ts
+++ after/lib/src/main/ets/format/msg.ts
@@ -1,25 +1,25 @@
-            }
-          }
-        } else if (entry.name.startsWith("__nameid_version1.0")) {
-          // const buff = this._cfb.readStorage(entry);
-          // for (const entry of buff) {
-          //   logger.trace("__nameid_0", `__nameid name:${entry.name}`);
-          //   if (entry.name.startsWith("__substg1.0_")) {
-          //     let pa: DirectoryItem = this.parseDirectory(entry);
-          //     logger.trace("__nameid_1",
-          //       pa.name,
-          //       pa.value,
-          //       pa.typeCode.toString(16),
-          //       entry.startSectorLocation,
-          //       entry.streamSize
-          //     );
-          //   }
-          // }
-        }
-        break;
-      case DirectoryObjectType.RootStorage:
-        logger.trace(
-          "__root_storage",
-          ` name:${entry.name}`,
-          ` entry.startSectorLocation:${entry.startSectorLocation}`,
-          ` entry.streamSize:${entry.streamSize}`
+        );
+        break
+    }
+  }
+
+  toObject(): object {
+    this._cfb.parse()
+    const t = this.parseEntry(this._cfb.getRootEntry());
+    return t;
+  }
+
+  parseEntry(entry: DirectoryEntry) {
+    if (entry.objectType == DirectoryObjectType.Stream) {
+      const o = this.parseDirectory(entry);
+      return o;
+    } else {
+      const children = this._cfb.readStorage(entry).map(e => this.parseEntry(e));
+      const o = {
+        name: entry.name,
+        children
+      }
+      return o;
+    }
+  }
+
```

## 333297ce67f7
- project: `ohos_mail_base`
- round_fixed: `1` (early)
- rule: `security/no-commented-code` (warn)
- file: `lib/src/main/ets/utils/mime.ts`:142:3
- patch: shape=`modify`, changed_lines=3

### Before (local context)
```
   136      extraHeader,
   137      attachments: attachmentList,
   138      inlines: inlineList,
   139    };
   140    const res = await stringifyMail(mail, bufferCreator);
   141    return res;
   142    // return res.readAll();
   143  }
   144  
   145  export async function serializeMIME(
   146    headers: Array<[string, string]>,
   147    body: string,
   148    option: EncodeOption = {}
```

### After (local context)
```
   136      replyTo: [],
   137      extraHeader,
   138      attachments: attachmentList,
   139      inlines: inlineList,
   140    };
   141    const res = await stringifyMail(mail, bufferCreator);
   142    return res;
   143  }
   144  
   145  export async function serializeMIME(
   146    headers: Array<[string, string]>,
   147    body: string,
   148    option: EncodeOption = {}
```

### Local diff
```diff
--- before/lib/src/main/ets/utils/mime.ts
+++ after/lib/src/main/ets/utils/mime.ts
@@ -1,3 +1,4 @@
+  const mail: MailItem = {
     ...mailHead,
     date: new Date,
     messageId: '',
@@ -10,7 +11,6 @@
   };
   const res = await stringifyMail(mail, bufferCreator);
   return res;
-  // return res.readAll();
 }
 
 export async function serializeMIME(
```
