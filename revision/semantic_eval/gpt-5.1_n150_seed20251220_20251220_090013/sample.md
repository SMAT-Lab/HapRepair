# Repair semantic-eval sample (gpt-5.1)

- n=150, seed=20251220, max_round=5

## 0ffe7a5f5e5e
- project: `HealthyPotAssistant`
- round_fixed: `1` (early)
- rule: `performance/init-list-component` (warn)
- file: `entry/src/main/ets/MainAbility/pages/jobModelSettings.ets`:35:9
- patch: shape=`modify`, changed_lines=39

### Before (local context)
```
    29    }
    30  
    31    build() {
    32      Row() {
    33        Column() {
    34          MTopBar({titleStr: this.message})
    35          List(){
    36            ForEach(this.jobItems, (item: JobData, indexd: number) => {
    37              ListItem() {
    38                //TODO: 自定义组件不能正常显示 [phone][Ace       ERROR]  09/24 18:32:20 10492  [qjs_utils.cpp(JsStdDumpErrorAce)-(0)] [Engine Log] [DUMP] TypeError: cannot read property 'subscribeMe' of undefined
    39                JobListItem({itIndex: indexd, jobData: item, workStatus: this.workStatus, currentSelectId: $currentSelectId})
    40  //              TestSubItem({jobData: item, itIndex: indexd})
    41  //              Column() {
```

### After (local context)
```
    29    }
    30  
    31    build() {
    32      Row() {
    33        Column() {
    34          MTopBar({titleStr: this.message})
    35          List().width('100%').height('100%'){
    36            ForEach(this.jobItems, (item: JobData, indexd: number) => {
    37              ListItem() {
    38                //TODO: 自定义组件不能正常显示 [phone][Ace       ERROR]  09/24 18:32:20 10492  [qjs_utils.cpp(JsStdDumpErrorAce)-(0)] [Engine Log] [DUMP] TypeError: cannot read property 'subscribeMe' of undefined
    39                JobListItem({itIndex: indexd, jobData: item, workStatus: this.workStatus, currentSelectId: $currentSelectId})
    40  //              Column() {
    41  //                Row () {
```

### Local diff
```diff
--- before/entry/src/main/ets/MainAbility/pages/jobModelSettings.ets
+++ after/entry/src/main/ets/MainAbility/pages/jobModelSettings.ets
@@ -10,16 +10,16 @@
     Row() {
       Column() {
         MTopBar({titleStr: this.message})
-        List(){
+        List().width('100%').height('100%'){
           ForEach(this.jobItems, (item: JobData, indexd: number) => {
             ListItem() {
               //TODO: 自定义组件不能正常显示 [phone][Ace       ERROR]  09/24 18:32:20 10492  [qjs_utils.cpp(JsStdDumpErrorAce)-(0)] [Engine Log] [DUMP] TypeError: cannot read property 'subscribeMe' of undefined
               JobListItem({itIndex: indexd, jobData: item, workStatus: this.workStatus, currentSelectId: $currentSelectId})
-//              TestSubItem({jobData: item, itIndex: indexd})
 //              Column() {
 //                Row () {
-//                  Radio({ value: 'Radio'+(index + 1), group: 'radioGroup' })
 //                    .checked(item.id == this.currentSelectId)
 //                    .height(20)
 //                    .width(20)
 //                    .enabled(this.workStatus == 0)
+//                    .onChange((isChecked: boolean) => {
+//                      if (isChecked) {
```

## 7d01a8656dee
- project: `Image`
- round_fixed: `1` (early)
- rule: `performance/monitor-invisible-area-in-image-animation` (warn)
- file: `entry/src/main/ets/pages/ImageInterface/image_overlay.ets`:444:7
- patch: shape=`modify`, changed_lines=39

### Before (local context)
```
   438  @Component
   439  struct ModifierComponent {
   440    @Link customModifier: ImageAnimatorModifier
   441  
   442    build() {
   443      Column({ space: 10 }) {
   444        ImageAnimator()
   445          .width(200)
   446          .height(150)
   447          .attributeModifier(this.customModifier as CustomModifier)
   448      }
   449    }
   450  }
```

### After (local context)
```
   438  @Builder
   439  function MyCustomComponentBuilder(testData: TestAttributes) {
   440    Column({ space: 10 }) {
   441      ImageAnimator()
   442        .width(200)
   443        .height(150)
   444        .iterations(testData.iterations)
   445        .fillMode(testData.fillMode)
   446        .fixedSize(testData.fixedSize)
   447        .reverse(testData.reverse)
   448        .duration(testData.duration)
   449        .state(testData.state)
   450        .images(testData.images)
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/ImageInterface/image_overlay.ets
+++ after/entry/src/main/ets/pages/ImageInterface/image_overlay.ets
@@ -1,3 +1,19 @@
+      .reverse(testData.reverse)
+      .duration(testData.duration)
+      .state(testData.state)
+      .images(testData.images)
+  }
+}
+@Builder
+function MyCustomComponentBuilder(testData: TestAttributes) {
+  Column({ space: 10 }) {
+    ImageAnimator()
+      .width(200)
+      .height(150)
+      .iterations(testData.iterations)
+      .fillMode(testData.fillMode)
+      .fixedSize(testData.fixedSize)
+      .reverse(testData.reverse)
       .duration(testData.duration)
       .state(testData.state)
       .images(testData.images)
@@ -7,19 +23,3 @@
 @Component
 struct ModifierComponent {
   @Link customModifier: ImageAnimatorModifier
-
-  build() {
-    Column({ space: 10 }) {
-      ImageAnimator()
-        .width(200)
-        .height(150)
-        .attributeModifier(this.customModifier as CustomModifier)
-    }
-  }
-}
-
-@Component
-struct MyCustomComponent {
-  @Prop testData: TestAttributes
-
-  build() {
```

## eea9159e398a
- project: `Image`
- round_fixed: `1` (early)
- rule: `performance/multiple-associations-state-var-check` (suggestion)
- file: `entry/src/main/ets/pages/ImageInterface/image_overlay.ets`:454:9
- patch: shape=`modify`, changed_lines=39

### Before (local context)
```
   448      }
   449    }
   450  }
   451  
   452  @Component
   453  struct MyCustomComponent {
   454    @Prop testData: TestAttributes
   455  
   456    build() {
   457      Column({ space: 10 }) {
   458        ImageAnimator()
   459          .width(200)
   460          .height(150)
```

### After (local context)
```
   448        .duration(testData.duration)
   449        .state(testData.state)
   450        .images(testData.images)
   451    }
   452  }
   453  
   454  @Component
   455  struct ModifierComponent {
   456    @Link customModifier: ImageAnimatorModifier
   457  
   458    build() {
   459      Column({ space: 10 }) {
   460        ImageAnimator()
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/ImageInterface/image_overlay.ets
+++ after/entry/src/main/ets/pages/ImageInterface/image_overlay.ets
@@ -1,3 +1,19 @@
+      .width(200)
+      .height(150)
+      .iterations(testData.iterations)
+      .fillMode(testData.fillMode)
+      .fixedSize(testData.fixedSize)
+      .reverse(testData.reverse)
+      .duration(testData.duration)
+      .state(testData.state)
+      .images(testData.images)
+  }
+}
+
+@Component
+struct ModifierComponent {
+  @Link customModifier: ImageAnimatorModifier
+
   build() {
     Column({ space: 10 }) {
       ImageAnimator()
@@ -7,19 +23,3 @@
     }
   }
 }
-
-@Component
-struct MyCustomComponent {
-  @Prop testData: TestAttributes
-
-  build() {
-    Column({ space: 10 }) {
-      ImageAnimator()
-        .width(200)
-        .height(150)
-        .iterations(this.testData.iterations)
-        .fillMode(this.testData.fillMode)
-        .fixedSize(this.testData.fixedSize)
-        .reverse(this.testData.reverse)
-        .duration(this.testData.duration)
-        .state(this.testData.state)
```

## ea4f8400c0bf
- project: `Image`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-redundant-state-var` (suggestion)
- file: `entry/src/main/ets/pages/ImageInterface/orientation.ets`:20:3
- patch: shape=`delete`, changed_lines=9

### Before (local context)
```
    14   */
    15  @Entry
    16  @Component
    17  struct ImageRotateOrientationIndex {
    18    @State message: string = 'Hello World';
    19    @State text: string = "Image组件引入exif规范不支持的图源，设置autoResize属性"
    20    @State orientation: ImageRotateOrientation = ImageRotateOrientation.UP
    21    @State autoResize: boolean = false
    22    @State resource: Resource = $r('app.media.huawei_200')
    23    @State onTree: boolean = true
    24    @State rotateDegree: number = 0
    25    @State fit: ImageFit = ImageFit.None
    26    res: Resource = $r('app.media.UP')
```

### After (local context)
```
    14   */
    15  @Entry
    16  @Component
    17  struct ImageRotateOrientationIndex {
    18    res: Resource = $r('app.media.UP')
    19    idx: number = 0
    20    idx2: number = 0
    21    orientations: Array<ImageRotateOrientation> = [
    22      ImageRotateOrientation.DOWN,
    23      ImageRotateOrientation.LEFT,
    24      ImageRotateOrientation.RIGHT,
    25      ImageRotateOrientation.AUTO,
    26      ImageRotateOrientation.UP,
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/ImageInterface/orientation.ets
+++ after/entry/src/main/ets/pages/ImageInterface/orientation.ets
@@ -8,14 +8,6 @@
 @Entry
 @Component
 struct ImageRotateOrientationIndex {
-  @State message: string = 'Hello World';
-  @State text: string = "Image组件引入exif规范不支持的图源，设置autoResize属性"
-  @State orientation: ImageRotateOrientation = ImageRotateOrientation.UP
-  @State autoResize: boolean = false
-  @State resource: Resource = $r('app.media.huawei_200')
-  @State onTree: boolean = true
-  @State rotateDegree: number = 0
-  @State fit: ImageFit = ImageFit.None
   res: Resource = $r('app.media.UP')
   idx: number = 0
   idx2: number = 0
@@ -23,3 +15,11 @@
     ImageRotateOrientation.DOWN,
     ImageRotateOrientation.LEFT,
     ImageRotateOrientation.RIGHT,
+    ImageRotateOrientation.AUTO,
+    ImageRotateOrientation.UP,
+  ]
+  repeats: Array<ImageRepeat> = [
+    ImageRepeat.NoRepeat,
+    ImageRepeat.X,
+    ImageRepeat.Y,
+    ImageRepeat.XY,
```

## 0ec8c28aa78e
- project: `Image`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-redundant-state-var` (suggestion)
- file: `entry/src/main/ets/pages/ImageInterface/orientation.ets`:24:3
- patch: shape=`delete`, changed_lines=9

### Before (local context)
```
    18    @State message: string = 'Hello World';
    19    @State text: string = "Image组件引入exif规范不支持的图源，设置autoResize属性"
    20    @State orientation: ImageRotateOrientation = ImageRotateOrientation.UP
    21    @State autoResize: boolean = false
    22    @State resource: Resource = $r('app.media.huawei_200')
    23    @State onTree: boolean = true
    24    @State rotateDegree: number = 0
    25    @State fit: ImageFit = ImageFit.None
    26    res: Resource = $r('app.media.UP')
    27    idx: number = 0
    28    idx2: number = 0
    29    orientations: Array<ImageRotateOrientation> = [
    30      ImageRotateOrientation.DOWN,
```

### After (local context)
```
    18    res: Resource = $r('app.media.UP')
    19    idx: number = 0
    20    idx2: number = 0
    21    orientations: Array<ImageRotateOrientation> = [
    22      ImageRotateOrientation.DOWN,
    23      ImageRotateOrientation.LEFT,
    24      ImageRotateOrientation.RIGHT,
    25      ImageRotateOrientation.AUTO,
    26      ImageRotateOrientation.UP,
    27    ]
    28    repeats: Array<ImageRepeat> = [
    29      ImageRepeat.NoRepeat,
    30      ImageRepeat.X,
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/ImageInterface/orientation.ets
+++ after/entry/src/main/ets/pages/ImageInterface/orientation.ets
@@ -4,14 +4,6 @@
 @Entry
 @Component
 struct ImageRotateOrientationIndex {
-  @State message: string = 'Hello World';
-  @State text: string = "Image组件引入exif规范不支持的图源，设置autoResize属性"
-  @State orientation: ImageRotateOrientation = ImageRotateOrientation.UP
-  @State autoResize: boolean = false
-  @State resource: Resource = $r('app.media.huawei_200')
-  @State onTree: boolean = true
-  @State rotateDegree: number = 0
-  @State fit: ImageFit = ImageFit.None
   res: Resource = $r('app.media.UP')
   idx: number = 0
   idx2: number = 0
@@ -22,4 +14,12 @@
     ImageRotateOrientation.AUTO,
     ImageRotateOrientation.UP,
   ]
-  @State repeat: ImageRepeat = ImageRepeat.NoRepeat
+  repeats: Array<ImageRepeat> = [
+    ImageRepeat.NoRepeat,
+    ImageRepeat.X,
+    ImageRepeat.Y,
+    ImageRepeat.XY,
+  ]
+
+  build() {
+    Column() {
```

## 123d58adb8f2
- project: `Image`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-redundant-state-var` (suggestion)
- file: `entry/src/main/ets/pages/ImageOrientation/test10.ets`:62:3
- patch: shape=`modify`, changed_lines=17

### Before (local context)
```
    56    @State res6: ResourceStr | undefined = $r('app.media.huawei_200')
    57    @State text4: string = "Image组件引入exif规范支持的图源，设置borderRadius属性"
    58    @State orientation: ImageRotateOrientation = ImageRotateOrientation.AUTO
    59    @State fit: ImageFit = ImageFit.Cover
    60    @State repeat: ImageRepeat = ImageRepeat.NoRepeat
    61    @State autoResize: boolean = false
    62    @State modifier: OrientationModifier = new OrientationModifier()
    63    @State deleteOnTree: boolean = true
    64    @State left: number = 10
    65    @State right: number = 10
    66    @State top: number = 20
    67    @State bottom: number = 20
    68  
```

### After (local context)
```
    56    @State res5: ResourceStr | undefined = $r('app.media.leaf')
    57    @State res6: ResourceStr | undefined = $r('app.media.huawei_200')
    58    // 直接使用一般变量即可
    59    text4: string = "Image组件引入exif规范支持的图源，设置borderRadius属性"
    60    @State orientation: ImageRotateOrientation = ImageRotateOrientation.AUTO
    61    fit: ImageFit = ImageFit.Cover
    62    @State repeat: ImageRepeat = ImageRepeat.NoRepeat
    63    @State autoResize: boolean = false
    64    modifier: OrientationModifier = new OrientationModifier()
    65    // 直接使用一般变量即可
    66    deleteOnTree: boolean = true
    67    // 直接使用一般变量即可
    68    left: number = 10
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/ImageOrientation/test10.ets
+++ after/entry/src/main/ets/pages/ImageOrientation/test10.ets
@@ -1,25 +1,25 @@
+@Component
 struct Test01 {
-  @State res1: ResourceStr | undefined = $r('app.media.horizontal')
+  res1: ResourceStr | undefined = $r('app.media.horizontal')
   @State res2: ResourceStr | undefined = $r('app.media.rotate90cw')
   @State res3: ResourceStr | undefined = $r('app.media.rotate270cw')
-  @State res4: ResourceStr | undefined = $r('app.media.rotate180')
+  res4: ResourceStr | undefined = $r('app.media.rotate180')
   @State res5: ResourceStr | undefined = $r('app.media.leaf')
   @State res6: ResourceStr | undefined = $r('app.media.huawei_200')
-  @State text4: string = "Image组件引入exif规范支持的图源，设置borderRadius属性"
+  // 直接使用一般变量即可
+  text4: string = "Image组件引入exif规范支持的图源，设置borderRadius属性"
   @State orientation: ImageRotateOrientation = ImageRotateOrientation.AUTO
-  @State fit: ImageFit = ImageFit.Cover
+  fit: ImageFit = ImageFit.Cover
   @State repeat: ImageRepeat = ImageRepeat.NoRepeat
   @State autoResize: boolean = false
-  @State modifier: OrientationModifier = new OrientationModifier()
-  @State deleteOnTree: boolean = true
-  @State left: number = 10
-  @State right: number = 10
-  @State top: number = 20
-  @State bottom: number = 20
-
-
-  idx: number = 0
-  idx2: number = 0
-  orientations: Array<ImageRotateOrientation> = [
-    ImageRotateOrientation.DOWN,
-    ImageRotateOrientation.LEFT,
+  modifier: OrientationModifier = new OrientationModifier()
+  // 直接使用一般变量即可
+  deleteOnTree: boolean = true
+  // 直接使用一般变量即可
+  left: number = 10
+  // 直接使用一般变量即可
+  right: number = 10
+  // 直接使用一般变量即可
+  top: number = 20
+  // 直接使用一般变量即可
+  bottom: number = 20
```

## d84a598965b5
- project: `Image`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-redundant-state-var` (suggestion)
- file: `entry/src/main/ets/pages/ImageOrientation/test11.ets`:22:3
- patch: shape=`modify`, changed_lines=2

### Before (local context)
```
    16  import { image } from '@kit.ImageKit'
    17  
    18  
    19  @Entry
    20  @Component
    21  struct Test01 {
    22    @State text: string = "Image组件引入exif规范不支持的图源，设置autoResize属性"
    23    @State orientation: ImageRotateOrientation = ImageRotateOrientation.UP
    24    @State autoResize: boolean = false
    25    @State resource: Resource = $r('app.media.huawei_200')
    26    @State gif: Resource = $r('app.media.leaf')
    27    @State onTree: boolean = true
    28    @State rotateDegree: number = 0
```

### After (local context)
```
    16  import { image } from '@kit.ImageKit'
    17  
    18  
    19  @Entry
    20  @Component
    21  struct Test01 {
    22    text: string = "Image组件引入exif规范不支持的图源，设置autoResize属性"
    23    @State orientation: ImageRotateOrientation = ImageRotateOrientation.UP
    24    @State autoResize: boolean = false
    25    @State resource: Resource = $r('app.media.huawei_200')
    26    gif: Resource = $r('app.media.leaf')
    27    @State onTree: boolean = true
    28    @State rotateDegree: number = 0
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/ImageOrientation/test11.ets
+++ after/entry/src/main/ets/pages/ImageOrientation/test11.ets
@@ -10,11 +10,11 @@
 @Entry
 @Component
 struct Test01 {
-  @State text: string = "Image组件引入exif规范不支持的图源，设置autoResize属性"
+  text: string = "Image组件引入exif规范不支持的图源，设置autoResize属性"
   @State orientation: ImageRotateOrientation = ImageRotateOrientation.UP
   @State autoResize: boolean = false
   @State resource: Resource = $r('app.media.huawei_200')
-  @State gif: Resource = $r('app.media.leaf')
+  gif: Resource = $r('app.media.leaf')
   @State onTree: boolean = true
   @State rotateDegree: number = 0
   @State fit: ImageFit = ImageFit.None
```

## b2e3336aad4c
- project: `Image`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-no-state-var-access-in-loop` (warn)
- file: `entry/src/main/ets/pages/ImageParallelization/ImageCallBack.ets`:594:17
- patch: shape=`modify`, changed_lines=157

### Before (local context)
```
   588              ForEach(forEachItems100,(item: Int, index:Double)=>{
   589                Column(undefined) {
   590                  Text("image").fontSize(40)
   591                  Text('---------------------------').fontSize(14)
   592                  Text('事件').fontSize(22).margin({ bottom: 15 } as Margin)
   593  
   594                  Image($r('app.media.startIcon'))
   595                    .width(this.imageWidth)
   596                    .height(200)
   597                    .objectFit(ImageFit.Contain)
   598                    .visibility(this.visible)
   599                    .onComplete(() => {
   600                      this.visible = Visibility.Visible;
```

### After (local context)
```
   588            ParallelizeUI({}as ParallelOption) {
   589              ForEach(forEachItems100,(item: Int, index:Double)=>{
   590                Column(undefined) {
   591                  Text("image").fontSize(40)
   592                  Text('---------------------------').fontSize(14)
   593                  Text('事件').fontSize(22).margin({ bottom: 15 } as Margin)
   594  
   595                  Image($r('app.media.startIcon'))
   596                    .width(this.imageWidth)
   597                    .height(200)
   598                    .objectFit(ImageFit.Contain)
   599                    .visibility(this.visible)
   600                    .onComplete(() => {
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/ImageParallelization/ImageCallBack.ets
+++ after/entry/src/main/ets/pages/ImageParallelization/ImageCallBack.ets
@@ -1,3 +1,4 @@
+                    this.src2 = this.imageOne;
                     console.info('Test onFinish')
                   })
               }
@@ -22,4 +23,3 @@
                   })
                   .onError(() => {
                     setTimeout(() => {
-                      this.visible = Visibility.Visible;
```

## 0db89afe4d0a
- project: `Image`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-redundant-state-var` (suggestion)
- file: `entry/src/main/ets/pages/example/ImageExample038.ets`:47:3
- patch: shape=`modify`, changed_lines=10

### Before (local context)
```
    41    }
    42  }
    43  
    44  @Entry
    45  @Component
    46  struct ImageExample038 {
    47    @State message: string = 'Hello World';
    48    @State text: string = 'Image组件引入exif规范不支持的图源，设置autoResize属性'
    49    @State orientation: ImageRotateOrientation = ImageRotateOrientation.UP
    50    @State autoResize: boolean = false
    51    @State resource: Resource = $r('app.media.image')
    52    @State onTree: boolean = true
    53    @State rotateDegree: number = 0
```

### After (local context)
```
    41      instance.orientation(this.selfOrientation1)
    42    }
    43  }
    44  
    45  @Entry
    46  @Component
    47  struct ImageExample038 {
    48    res: Resource = $r('app.media.UP')
    49    @State selfOrientation: ImageRotateOrientation = ImageRotateOrientation.AUTO;
    50    enabledModifier: boolean = true;
    51  
    52    build() {
    53      Column() {
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/example/ImageExample038.ets
+++ after/entry/src/main/ets/pages/example/ImageExample038.ets
@@ -1,3 +1,4 @@
+    super();
     console.log('hello MyImageModifier');
     this.selfOrientation1 = orientation;
   }
@@ -10,16 +11,15 @@
 @Entry
 @Component
 struct ImageExample038 {
-  @State message: string = 'Hello World';
-  @State text: string = 'Image组件引入exif规范不支持的图源，设置autoResize属性'
-  @State orientation: ImageRotateOrientation = ImageRotateOrientation.UP
-  @State autoResize: boolean = false
-  @State resource: Resource = $r('app.media.image')
-  @State onTree: boolean = true
-  @State rotateDegree: number = 0
-  @State fit: ImageFit = ImageFit.None
   res: Resource = $r('app.media.UP')
   @State selfOrientation: ImageRotateOrientation = ImageRotateOrientation.AUTO;
-  @State enabledModifier: boolean = true;
+  enabledModifier: boolean = true;
 
   build() {
+    Column() {
+      Column() {
+        Row() {
+          Button('AUTO').onClick(() => {
+            this.selfOrientation = ImageRotateOrientation.AUTO
+          })
+          Button('LEFT').onClick(() => {
```

## c4db066dc3f5
- project: `Image`
- round_fixed: `2` (early)
- rule: `performance/avoid-overusing-custom-component-check` (warn)
- file: `entry/src/main/ets/pages/ImageDfxExamples/ImageDfxExamples0010.ets`:110:8
- patch: shape=`modify`, changed_lines=45

### Before (local context)
```
   104  }
   105  
   106  /**
   107   * Demo 2: Load image from network URL
   108   */
   109  @Component
   110  struct NetworkImageDemo {
   111    // 直接使用一般变量即可
   112    url: string = 'https://xxx.jpg'
   113    @State reloadKey: number = 0
   114  
   115    build() {
   116      Column() {
```

### After (local context)
```
   104  
   105  /**
   106   * Demo 2: Load image from network URL
   107   */
   108  @Component
   109  struct NetworkImageDemo {
   110    // 直接使用一般变量即可
   111    url: string = 'https://xxx.jpg'
   112    @State reloadKey: number = 0
   113  
   114    build() {
   115      Column() {
   116        Text('Network image loading')
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/ImageDfxExamples/ImageDfxExamples0010.ets
+++ after/entry/src/main/ets/pages/ImageDfxExamples/ImageDfxExamples0010.ets
@@ -1,8 +1,7 @@
-      Image($r('app.media.img_1'))
-        .width(200)
-        .height(200)
-        .margin({ top: 20 })
-    }
+    Image($r('app.media.img_1'))
+      .width(200)
+      .height(200)
+      .margin({ top: 20 })
   }
 }
 
@@ -23,3 +22,4 @@
 
       Image(this.url + '?v=' + this.reloadKey)
         .width(200)
+        .height(200)
```

## d587951c6841
- project: `Image`
- round_fixed: `2` (early)
- rule: `performance/hp-arkui-no-state-var-access-in-loop` (warn)
- file: `entry/src/main/ets/pages/ImageParallelization/ImageCallBack.ets`:136:17
- patch: shape=`delete`, changed_lines=69

### Before (local context)
```
   130                      setTimeout(() => {
   131                        this.visible = Visibility.Visible;
   132                        this.moveImg.pop();
   133                        console.info('Test onError')
   134                      }, 2600)
   135                    })
   136                  Image(this.src2)
   137                    .width(100)
   138                    .height(100)
   139                    .onFinish(() => {
   140                      this.src2 = this.imageOne;
   141                      console.info('Test onFinish')
   142                    })
```

### After (local context)
```
   130                        this.visible = Visibility.Visible;
   131                        this.moveImg.pop();
   132                        console.info('Test onError')
   133                      }, 2600)
   134                    })
   135                  Image(this.src2)
   136                    .width(100)
   137                    .height(100)
   138                    .onFinish(() => {
   139                      this.src2 = this.imageOne;
   140                      console.info('Test onFinish')
   141                    })
   142                }
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/ImageParallelization/ImageCallBack.ets
+++ after/entry/src/main/ets/pages/ImageParallelization/ImageCallBack.ets
@@ -1,4 +1,3 @@
-                  .onComplete(() => {
                     this.visible = Visibility.Visible;
                     this.moveImg.pop();
                     console.info('Test onComplete')
@@ -23,3 +22,4 @@
           ParallelizeUI({}as ParallelOption) {
             ForEach(forEachItems100,(item: Int, index:Double)=>{
               Column(undefined) {
+                Text("image").fontSize(40)
```

## 00b03b41198a
- project: `Image`
- round_fixed: `2` (early)
- rule: `performance/hp-arkui-no-state-var-access-in-loop` (warn)
- file: `entry/src/main/ets/pages/ImageParallelization/ImageCallBack.ets`:140:21
- patch: shape=`delete`, changed_lines=69

### Before (local context)
```
   134                      }, 2600)
   135                    })
   136                  Image(this.src2)
   137                    .width(100)
   138                    .height(100)
   139                    .onFinish(() => {
   140                      this.src2 = this.imageOne;
   141                      console.info('Test onFinish')
   142                    })
   143                }
   144              })
   145            }
   146            ParallelizeUI({}as ParallelOption) {
```

### After (local context)
```
   134                    })
   135                  Image(this.src2)
   136                    .width(100)
   137                    .height(100)
   138                    .onFinish(() => {
   139                      this.src2 = this.imageOne;
   140                      console.info('Test onFinish')
   141                    })
   142                }
   143              })
   144            }
   145            ParallelizeUI({}as ParallelOption) {
   146              ForEach(forEachItems100,(item: Int, index:Double)=>{
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/ImageParallelization/ImageCallBack.ets
+++ after/entry/src/main/ets/pages/ImageParallelization/ImageCallBack.ets
@@ -1,4 +1,3 @@
-                  })
                   .onError(() => {
                     setTimeout(() => {
                       this.visible = Visibility.Visible;
@@ -23,3 +22,4 @@
                 Text('---------------------------').fontSize(14)
                 Text('事件').fontSize(22).margin({ bottom: 15 } as Margin)
 
+                Image($r('app.media.startIcon'))
```

## 544f7dac7d10
- project: `Image`
- round_fixed: `2` (early)
- rule: `performance/hp-arkui-no-state-var-access-in-loop` (warn)
- file: `entry/src/main/ets/pages/ImageParallelization/ImageCallBack.ets`:335:23
- patch: shape=`delete`, changed_lines=69

### Before (local context)
```
   329                      this.visible = Visibility.Visible;
   330                      this.moveImg.pop();
   331                      console.info('Test onComplete')
   332                    })
   333                    .onError(() => {
   334                      setTimeout(() => {
   335                        this.visible = Visibility.Visible;
   336                        this.moveImg.pop();
   337                        console.info('Test onError')
   338                      }, 2600)
   339                    })
   340                  Image(this.src2)
   341                    .width(100)
```

### After (local context)
```
   329                      this.moveImg.pop();
   330                      console.info('Test onComplete')
   331                    })
   332                    .onError(() => {
   333                      setTimeout(() => {
   334                        this.visible = Visibility.Visible;
   335                        this.moveImg.pop();
   336                        console.info('Test onError')
   337                      }, 2600)
   338                    })
   339                  Image(this.src2)
   340                    .width(100)
   341                    .height(100)
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/ImageParallelization/ImageCallBack.ets
+++ after/entry/src/main/ets/pages/ImageParallelization/ImageCallBack.ets
@@ -1,4 +1,3 @@
-                Image($r('app.media.startIcon'))
                   .width(this.imageWidth)
                   .height(200)
                   .objectFit(ImageFit.Contain)
@@ -23,3 +22,4 @@
                     console.info('Test onFinish')
                   })
               }
+            })
```

## c290009540f7
- project: `Image`
- round_fixed: `2` (early)
- rule: `performance/hp-arkui-no-state-var-access-in-loop` (warn)
- file: `entry/src/main/ets/pages/ImageParallelization/ImageCallBack.ets`:437:23
- patch: shape=`delete`, changed_lines=69

### Before (local context)
```
   431                      this.visible = Visibility.Visible;
   432                      this.moveImg.pop();
   433                      console.info('Test onComplete')
   434                    })
   435                    .onError(() => {
   436                      setTimeout(() => {
   437                        this.visible = Visibility.Visible;
   438                        this.moveImg.pop();
   439                        console.info('Test onError')
   440                      }, 2600)
   441                    })
   442                  Image(this.src2)
   443                    .width(100)
```

### After (local context)
```
   431                      this.moveImg.pop();
   432                      console.info('Test onComplete')
   433                    })
   434                    .onError(() => {
   435                      setTimeout(() => {
   436                        this.visible = Visibility.Visible;
   437                        this.moveImg.pop();
   438                        console.info('Test onError')
   439                      }, 2600)
   440                    })
   441                  Image(this.src2)
   442                    .width(100)
   443                    .height(100)
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/ImageParallelization/ImageCallBack.ets
+++ after/entry/src/main/ets/pages/ImageParallelization/ImageCallBack.ets
@@ -1,4 +1,3 @@
-                Image($r('app.media.startIcon'))
                   .width(this.imageWidth)
                   .height(200)
                   .objectFit(ImageFit.Contain)
@@ -23,3 +22,4 @@
                     console.info('Test onFinish')
                   })
               }
+            })
```

## 78f8f1fac507
- project: `Info`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-redundant-state-var` (suggestion)
- file: `entry/src/main/ets/pages/datapanel/DataPanelExample002.ets`:32:3
- patch: shape=`modify`, changed_lines=6

### Before (local context)
```
    26    public colorShadow2: LinearGradient =
    27      new LinearGradient([{ color: '#65e26709', offset: 0 }, { color: '#65efbd08', offset: 1 }])
    28    public colorShadow3: LinearGradient =
    29      new LinearGradient([{ color: '#6572B513', offset: 0 }, { color: '#6508efa6', offset: 1 }])
    30    public colorShadow4: LinearGradient =
    31      new LinearGradient([{ color: '#65ed08f5', offset: 0 }, { color: '#65ef0849', offset: 1 }])
    32    @State color3: string = '#00FF00'
    33    @State color4: string = '#20FF0000'
    34    @State bgColor: string = '#08182431'
    35    @State offsetX: number = 15
    36    @State offsetY: number = 15
    37    @State radius: number = 5
    38    @State colorArray: Array<LinearGradient | ResourceColor> = [this.color1, this.color2, this.color3, this.color4]
```

### After (local context)
```
    26    public colorShadow2: LinearGradient =
    27      new LinearGradient([{ color: '#65e26709', offset: 0 }, { color: '#65efbd08', offset: 1 }])
    28    public colorShadow3: LinearGradient =
    29      new LinearGradient([{ color: '#6572B513', offset: 0 }, { color: '#6508efa6', offset: 1 }])
    30    public colorShadow4: LinearGradient =
    31      new LinearGradient([{ color: '#65ed08f5', offset: 0 }, { color: '#65ef0849', offset: 1 }])
    32    color3: string = '#00FF00'
    33    color4: string = '#20FF0000'
    34    bgColor: string = '#08182431'
    35    offsetX: number = 15
    36    offsetY: number = 15
    37    radius: number = 5
    38    @State colorArray: Array<LinearGradient | ResourceColor> = [this.color1, this.color2, this.color3, this.color4]
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/datapanel/DataPanelExample002.ets
+++ after/entry/src/main/ets/pages/datapanel/DataPanelExample002.ets
@@ -10,12 +10,12 @@
     new LinearGradient([{ color: '#6572B513', offset: 0 }, { color: '#6508efa6', offset: 1 }])
   public colorShadow4: LinearGradient =
     new LinearGradient([{ color: '#65ed08f5', offset: 0 }, { color: '#65ef0849', offset: 1 }])
-  @State color3: string = '#00FF00'
-  @State color4: string = '#20FF0000'
-  @State bgColor: string = '#08182431'
-  @State offsetX: number = 15
-  @State offsetY: number = 15
-  @State radius: number = 5
+  color3: string = '#00FF00'
+  color4: string = '#20FF0000'
+  bgColor: string = '#08182431'
+  offsetX: number = 15
+  offsetY: number = 15
+  radius: number = 5
   @State colorArray: Array<LinearGradient | ResourceColor> = [this.color1, this.color2, this.color3, this.color4]
   @State shadowColorArray: Array<LinearGradient | ResourceColor> =
     [this.colorShadow1, this.colorShadow2, this.colorShadow3, this.colorShadow4]
```

## ee30588f6834
- project: `Info`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-unchanged-state-var` (suggestion)
- file: `entry/src/main/ets/pages/gauge/GaugeTestCase001.ets`:28:3
- patch: shape=`modify`, changed_lines=26

### Before (local context)
```
    22    @State rating: number = 0;
    23    @State address: string = '';
    24    @State fillColor: string = '#FF000000';
    25    @State name: string = 'Gauge';
    26    @State stepTips: string = '操作步骤：拖动下方滑动条' + '\n' + '预期结果：仪表盘的刻度发生改变';
    27    @State view: boolean = false;
    28    @State active: boolean = false;
    29    @State intervalNum: number = 0;
    30    @State start: boolean = false
    31    @State fromStart: boolean = true
    32    @State step: number = 50
    33    @State loop: number = 3
    34    @State mColor: Color = Color.Gray
```

### After (local context)
```
    22  struct gauge {
    23    rating: number = 0;
    24    address: string = '';
    25    fillColor: string = '#FF000000';
    26    // name 未发生变化，改为普通变量
    27    name: string = 'Gauge';
    28    // stepTips 未发生变化，改为普通变量
    29    stepTips: string = '操作步骤：拖动下方滑动条' + '\n' + '预期结果：仪表盘的刻度发生改变';
    30    @State view: boolean = false;
    31    active: boolean = false;
    32    intervalNum: number = 0;
    33    start: boolean = false
    34    fromStart: boolean = true
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/gauge/GaugeTestCase001.ets
+++ after/entry/src/main/ets/pages/gauge/GaugeTestCase001.ets
@@ -1,25 +1,25 @@
+
 import router from '@ohos.router';
 import promptAction from '@ohos.promptAction';
 
 @Entry
 @Component
 struct gauge {
-  @State rating: number = 0;
-  @State address: string = '';
-  @State fillColor: string = '#FF000000';
-  @State name: string = 'Gauge';
-  @State stepTips: string = '操作步骤：拖动下方滑动条' + '\n' + '预期结果：仪表盘的刻度发生改变';
+  rating: number = 0;
+  address: string = '';
+  fillColor: string = '#FF000000';
+  // name 未发生变化，改为普通变量
+  name: string = 'Gauge';
+  // stepTips 未发生变化，改为普通变量
+  stepTips: string = '操作步骤：拖动下方滑动条' + '\n' + '预期结果：仪表盘的刻度发生改变';
   @State view: boolean = false;
-  @State active: boolean = false;
-  @State intervalNum: number = 0;
-  @State start: boolean = false
-  @State fromStart: boolean = true
-  @State step: number = 50
-  @State loop: number = 3
-  @State mColor: Color = Color.Gray
+  active: boolean = false;
+  intervalNum: number = 0;
+  start: boolean = false
+  fromStart: boolean = true
+  step: number = 50
+  loop: number = 3
+  mColor: Color = Color.Gray
   @State value: number = 0
-  @State startAngle: number = 0
-  @State endAngle: number = 360
-  @State mWidth: number = 20
-
-  @Builder
+  // startAngle 未发生变化，改为普通变量
+  startAngle: number = 0
```

## c9f9576382cd
- project: `Info`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-unchanged-state-var` (suggestion)
- file: `entry/src/main/ets/pages/gauge/GaugeTestCase004.ets`:22:3
- patch: shape=`modify`, changed_lines=6

### Before (local context)
```
    16  @Entry
    17  @Component
    18  struct GaugeLevel0_2 {
    19    @State value: number = 50;
    20    @State gaugeSize: number[] = [1, 100];
    21    @State angles: number[] = [210, 120, 150];
    22    @State strokeWidth: number = 5;
    23    @State color2: LinearGradient = new LinearGradient([
    24      { color: Color.Yellow, offset: 0 },
    25      { color: Color.Red, offset: 1 }
    26    ]);
    27    @State descriptionStr: string = 'AA';
    28    @State gaugeType1Size: number[] = [128, 80];
```

### After (local context)
```
    16  @Entry
    17  @Component
    18  struct GaugeLevel0_2 {
    19    // value 未发生变化，改为普通变量
    20    value: number = 50;
    21    @State gaugeSize: number[] = [1, 100];
    22    @State angles: number[] = [210, 120, 150];
    23    // strokeWidth 未发生变化，改为普通变量
    24    strokeWidth: number = 5;
    25    @State color2: LinearGradient = new LinearGradient([
    26      { color: Color.Yellow, offset: 0 },
    27      { color: Color.Red, offset: 1 }
    28    ]);
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/gauge/GaugeTestCase004.ets
+++ after/entry/src/main/ets/pages/gauge/GaugeTestCase004.ets
@@ -7,19 +7,19 @@
 @Entry
 @Component
 struct GaugeLevel0_2 {
-  @State value: number = 50;
+  // value 未发生变化，改为普通变量
+  value: number = 50;
   @State gaugeSize: number[] = [1, 100];
   @State angles: number[] = [210, 120, 150];
-  @State strokeWidth: number = 5;
+  // strokeWidth 未发生变化，改为普通变量
+  strokeWidth: number = 5;
   @State color2: LinearGradient = new LinearGradient([
     { color: Color.Yellow, offset: 0 },
     { color: Color.Red, offset: 1 }
   ]);
-  @State descriptionStr: string = 'AA';
+  // descriptionStr 未发生变化，改为普通变量
+  descriptionStr: string = 'AA';
   @State gaugeType1Size: number[] = [128, 80];
   @State gaugeType2Size: number[] = [80, 80];
   @State text: string | undefined = undefined;
   @State showText: boolean = false;
-  @State index: number = 0;
-  private testCases1: GaugeShadowOptions[] = [
-  // 0
```

## 823af37ac938
- project: `Info`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-redundant-state-var` (suggestion)
- file: `entry/src/main/ets/pages/textclock/TextClock-fontWeight.ets`:21:3
- patch: shape=`delete`, changed_lines=2

### Before (local context)
```
    15  
    16  import ConfigurationConstant from '@ohos.app.ability.ConfigurationConstant';
    17  import common from '@ohos.app.ability.common';
    18  @Entry
    19  @Component
    20  struct Page {
    21    @State message: string = 'Hello World';
    22    @State value: number = 0;
    23    @State isDark: boolean = false;
    24    @State context: common.UIAbilityContext = getContext(this) as common.UIAbilityContext
    25    build() {
    26      Column() {
    27        Row() {
```

### After (local context)
```
    15  
    16  import ConfigurationConstant from '@ohos.app.ability.ConfigurationConstant';
    17  import common from '@ohos.app.ability.common';
    18  @Entry
    19  @Component
    20  struct Page {
    21    @State isDark: boolean = false;
    22    @State context: common.UIAbilityContext = getContext(this) as common.UIAbilityContext
    23    build() {
    24      Column() {
    25        Row() {
    26          Text('TextClock-fontWeight字体粗细').fontSize(22).fontColor(0x000020).fontWeight(FontWeight.Bold)
    27        }
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/textclock/TextClock-fontWeight.ets
+++ after/entry/src/main/ets/pages/textclock/TextClock-fontWeight.ets
@@ -10,8 +10,6 @@
 @Entry
 @Component
 struct Page {
-  @State message: string = 'Hello World';
-  @State value: number = 0;
   @State isDark: boolean = false;
   @State context: common.UIAbilityContext = getContext(this) as common.UIAbilityContext
   build() {
@@ -23,3 +21,5 @@
       .width('95%')
       .borderRadius(15)
       .backgroundColor('#ffa9cbd6',)
+      .justifyContent(FlexAlign.Center)
+      Button('Change')
```

## 1261b02b3301
- project: `JS_dialog_box_static`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-unchanged-state-var` (suggestion)
- file: `entry/src/main/ets/pages/WebAttributeTest.ets`:75:3
- patch: shape=`modify`, changed_lines=26

### Before (local context)
```
    69    @State webCreated: boolean = false
    70    @State searchResultCount: number = -1
    71    @State onInterceptKeyEventCode: number = 0;
    72    @State onFaviconReceivedCalled: boolean = false
    73    @State outputStr: string = ''
    74    @State handleCancel: boolean = false;
    75    @State title: string = ''
    76    @State onRequestSelectedCalled: boolean = false
    77    @State testConsole: boolean = false
    78  
    79    aboutToAppear() {
    80      console.info('Entry aboutToAppear');
    81      let valueChangeEvent: emitter.InnerEvent = {
```

### After (local context)
```
    69    webInit: boolean = false
    70    webCreated: boolean = false
    71    searchResultCount: number = -1
    72    onInterceptKeyEventCode: number = 0;
    73    onFaviconReceivedCalled: boolean = false
    74    outputStr: string = ''
    75    handleCancel: boolean = false;
    76    title: string = ''
    77    onRequestSelectedCalled: boolean = false
    78    testConsole: boolean = false
    79  
    80    aboutToAppear() {
    81      console.info('Entry aboutToAppear');
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/WebAttributeTest.ets
+++ after/entry/src/main/ets/pages/WebAttributeTest.ets
@@ -1,18 +1,19 @@
-  @State httpErrorReceive: number = 0
-  @State isDownloadStart: boolean = false
-  @State newUrl: string = ""
-  @State isRefreshed: boolean = false
-  @State loadedUrl: string = ""
-  @State webInit: boolean = false
-  @State webCreated: boolean = false
-  @State searchResultCount: number = -1
-  @State onInterceptKeyEventCode: number = 0;
-  @State onFaviconReceivedCalled: boolean = false
-  @State outputStr: string = ''
-  @State handleCancel: boolean = false;
-  @State title: string = ''
-  @State onRequestSelectedCalled: boolean = false
-  @State testConsole: boolean = false
+  isLargeThan: boolean = false
+  httpErrorReceive: number = 0
+  isDownloadStart: boolean = false
+  newUrl: string = ""
+  isRefreshed: boolean = false
+  loadedUrl: string = ""
+  webInit: boolean = false
+  webCreated: boolean = false
+  searchResultCount: number = -1
+  onInterceptKeyEventCode: number = 0;
+  onFaviconReceivedCalled: boolean = false
+  outputStr: string = ''
+  handleCancel: boolean = false;
+  title: string = ''
+  onRequestSelectedCalled: boolean = false
+  testConsole: boolean = false
 
   aboutToAppear() {
     console.info('Entry aboutToAppear');
@@ -22,4 +23,3 @@
     }
     emitter.on(valueChangeEvent, this.valueChangeCallBack)
     console.info('Finish aboutToAppear ' + this.valueChangeCallBack);
-  }
```

## 0f601563d688
- project: `PullLinking`
- round_fixed: `1` (early)
- rule: `performance/init-list-component` (warn)
- file: `entry/src/main/ets/pages/OpenAppPage2.ets`:43:7
- patch: shape=`modify`, changed_lines=2

### Before (local context)
```
    37        hilog.error(0x0000, 'testTag', 'canOpenLink failed: %{public}s', message);
    38      }
    39    // [StartExclude can_open]
    40    }
    41    build() {
    42      Column() {
    43        List({ initialIndex: 0 }) {
    44          ListItem() {
    45            Row() {
    46              Button('MyAbility')
    47                .onClick(() => {
    48                })
    49            }
```

### After (local context)
```
    37        hilog.error(0x0000, 'testTag', 'canOpenLink failed: %{public}s', message);
    38      }
    39    // [StartExclude can_open]
    40    }
    41    build() {
    42      Column() {
    43        List({ initialIndex: 0 }).width('100%').height('100%') {
    44          ListItem() {
    45            Row() {
    46              Button('MyAbility')
    47                .onClick(() => {
    48                 this.handleScrollEvent()
    49                })
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/OpenAppPage2.ets
+++ after/entry/src/main/ets/pages/OpenAppPage2.ets
@@ -10,11 +10,12 @@
   }
   build() {
     Column() {
-      List({ initialIndex: 0 }) {
+      List({ initialIndex: 0 }).width('100%').height('100%') {
         ListItem() {
           Row() {
             Button('MyAbility')
               .onClick(() => {
+               this.handleScrollEvent()
               })
           }
         }
@@ -22,4 +23,3 @@
     }
   }
 }
-// [EndExclude can_open]
```

## 69b106590875
- project: `TextComponentTest`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-unchanged-state-var` (suggestion)
- file: `entry/src/main/ets/pages/02-TextSample/02-textNumberTransition.ets`:152:3
- patch: shape=`modify`, changed_lines=13

### Before (local context)
```
   146    @State wordBreakOptions: WordBreak[] =
   147      [WordBreak.NORMAL, WordBreak.BREAK_ALL, WordBreak.BREAK_WORD, WordBreak.HYPHENATION]
   148    @State clip1: boolean = true;
   149    @State start: number = 0;
   150    @State end: number = 1;
   151    @State caretColor: Color = Color.Blue
   152    @State padding1: number = 10
   153    @State enableDataDetector: boolean = true
   154    @State enableprevieew: boolean | undefined | null = true
   155    scrollerForList: Scroller = new Scroller();
   156    @State visi: Visibility = Visibility.Visible
   157  
   158    @Builder
```

### After (local context)
```
   146    @State end: number = 1;
   147    @State caretColor: Color = Color.Blue
   148    // padding1 未发生变化，改为普通变量
   149    padding1: number = 10
   150    // enableDataDetector 未发生变化，改为普通变量
   151    enableDataDetector: boolean = true
   152    @State enableprevieew: boolean | undefined | null = true
   153    scrollerForList: Scroller = new Scroller();
   154    @State visi: Visibility = Visibility.Visible
   155  
   156    @Builder
   157    MyMenu() {
   158  
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/02-TextSample/02-textNumberTransition.ets
+++ after/entry/src/main/ets/pages/02-TextSample/02-textNumberTransition.ets
@@ -1,7 +1,3 @@
-      repeating: true,
-    };
-  @State textOverflowIndex: number = 0;
-  @State textOverflow: TextOverflow[] =
     [TextOverflow.None, TextOverflow.MARQUEE, TextOverflow.Ellipsis];
   @State wordBreakIndex: number = 0;
   @State wordBreakOptions: WordBreak[] =
@@ -10,16 +6,20 @@
   @State start: number = 0;
   @State end: number = 1;
   @State caretColor: Color = Color.Blue
-  @State padding1: number = 10
-  @State enableDataDetector: boolean = true
+  // padding1 未发生变化，改为普通变量
+  padding1: number = 10
+  // enableDataDetector 未发生变化，改为普通变量
+  enableDataDetector: boolean = true
   @State enableprevieew: boolean | undefined | null = true
   scrollerForList: Scroller = new Scroller();
   @State visi: Visibility = Visibility.Visible
 
   @Builder
   MyMenu() {
-    Column() {
-      Text("hhh")
-    }
+
+    Text("hhh")
+    
   }
 
+  build() {
+    Row() {
```

## 029c286939be
- project: `TextComponentTest`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-unchanged-state-var` (suggestion)
- file: `entry/src/main/ets/pages/04-UseText/03-Symbol/10-ReplaceSymbolEffect.ets`:20:3
- patch: shape=`modify`, changed_lines=3

### Before (local context)
```
    14   */
    15  
    16  @Entry
    17  @Component
    18  struct ReplaceSymbolEffect {
    19    @State triggerValueReplace: number = 0;
    20    @State renderMode: number = 1;
    21    replaceFlag: boolean = true;
    22  
    23    build() {
    24      Column() {
    25        Text('禁用动效')
    26        SymbolGlyph(this.replaceFlag ? $r('sys.symbol.eye_slash') : $r('sys.symbol.eye'))
```

### After (local context)
```
    14   * limitations under the License.
    15   */
    16  
    17  @Entry
    18  @Component
    19  struct ReplaceSymbolEffect {
    20    @State triggerValueReplace: number = 0;
    21    // 直接使用一般变量即可
    22    renderMode: number = 1;
    23    replaceFlag: boolean = true;
    24  
    25    build() {
    26      Column() {
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/04-UseText/03-Symbol/10-ReplaceSymbolEffect.ets
+++ after/entry/src/main/ets/pages/04-UseText/03-Symbol/10-ReplaceSymbolEffect.ets
@@ -1,3 +1,4 @@
+ *     http://www.apache.org/licenses/LICENSE-2.0
  *
  * Unless required by applicable law or agreed to in writing, software
  * distributed under the License is distributed on an 'AS IS' BASIS,
@@ -10,7 +11,8 @@
 @Component
 struct ReplaceSymbolEffect {
   @State triggerValueReplace: number = 0;
-  @State renderMode: number = 1;
+  // 直接使用一般变量即可
+  renderMode: number = 1;
   replaceFlag: boolean = true;
 
   build() {
@@ -21,5 +23,3 @@
         .renderingStrategy(this.renderMode)
         .symbolEffect(new ReplaceSymbolEffect(EffectScope.LAYER,
           ReplaceEffectType.SLASH_OVERLAY), this.triggerValueReplace)
-      Button('trigger').onClick(() => {
-        this.replaceFlag = !this.replaceFlag;
```

## 438100284307
- project: `TextComponentTest`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-redundant-state-var` (suggestion)
- file: `entry/src/main/ets/pages/20-dev/TextInput.ets`:48:3
- patch: shape=`modify`, changed_lines=23

### Before (local context)
```
    42    @State lineHeight: (number | string | Resource)[] = [-5, 0, 10, 20, '40vp', '30%']
    43    @State lineHeightIndex: number = 0
    44    @State heightTest: number = 80
    45    @State widthTest: number = 250
    46    @State paddingTest: number = 0
    47    @State wordBreak: WordBreak[] = [WordBreak.NORMAL, WordBreak.BREAK_ALL, WordBreak.BREAK_WORD]
    48    @State wordBreakIndex: number = 0
    49    @State sizeAll: (number | string | Resource | undefined)[] = [-1, 0, 20, 'hsp', undefined, '20px', '10abc', $r('app.string.number_20px'), $r('app.string.number_10abc')]
    50    @State maxFontSize: number = 2
    51    @State minFontSize: number = 2
    52    @State heightAdaptivePolicy: TextHeightAdaptivePolicy[] = [TextHeightAdaptivePolicy.MAX_LINES_FIRST, TextHeightAdaptivePolicy.MIN_FONT_SIZE_FIRST, TextHeightAdaptivePolicy.LAYOUT_CONSTRAINT_FIRST]
    53    @State heightAdaptiveIndex: number = 0
    54    @State selectionStart: number = 0
```

### After (local context)
```
    42    letterSpacingIndex: number = 4
    43    lineHeight: (number | string | Resource)[] = [-5, 0, 10, 20, '40vp', '30%']
    44    lineHeightIndex: number = 0
    45    heightTest: number = 80
    46    widthTest: number = 250
    47    paddingTest: number = 0
    48    wordBreak: WordBreak[] = [WordBreak.NORMAL, WordBreak.BREAK_ALL, WordBreak.BREAK_WORD]
    49    wordBreakIndex: number = 0
    50    sizeAll: (number | string | Resource | undefined)[] = [-1, 0, 20, 'hsp', undefined, '20px', '10abc', $r('app.string.number_20px'), $r('app.string.number_10abc')]
    51    maxFontSize: number = 2
    52    minFontSize: number = 2
    53    heightAdaptivePolicy: TextHeightAdaptivePolicy[] = [TextHeightAdaptivePolicy.MAX_LINES_FIRST, TextHeightAdaptivePolicy.MIN_FONT_SIZE_FIRST, TextHeightAdaptivePolicy.LAYOUT_CONSTRAINT_FIRST]
    54    heightAdaptiveIndex: number = 0
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/20-dev/TextInput.ets
+++ after/entry/src/main/ets/pages/20-dev/TextInput.ets
@@ -1,25 +1,25 @@
+  @State textAlignStr: string[] = ['Start', 'Center', 'End']
   @State textAlignIndex: number = 0
-  @State fontSize:number = 16
-  @State showUnderline: boolean = false
-  @State maxLength: number = 6
-  @State letterSpacing: (number | string | Resource)[] = [-2, 0, 3, '5px', '10%']
-  @State letterSpacingIndex: number = 4
-  @State lineHeight: (number | string | Resource)[] = [-5, 0, 10, 20, '40vp', '30%']
-  @State lineHeightIndex: number = 0
-  @State heightTest: number = 80
-  @State widthTest: number = 250
-  @State paddingTest: number = 0
-  @State wordBreak: WordBreak[] = [WordBreak.NORMAL, WordBreak.BREAK_ALL, WordBreak.BREAK_WORD]
-  @State wordBreakIndex: number = 0
-  @State sizeAll: (number | string | Resource | undefined)[] = [-1, 0, 20, 'hsp', undefined, '20px', '10abc', $r('app.string.number_20px'), $r('app.string.number_10abc')]
-  @State maxFontSize: number = 2
-  @State minFontSize: number = 2
-  @State heightAdaptivePolicy: TextHeightAdaptivePolicy[] = [TextHeightAdaptivePolicy.MAX_LINES_FIRST, TextHeightAdaptivePolicy.MIN_FONT_SIZE_FIRST, TextHeightAdaptivePolicy.LAYOUT_CONSTRAINT_FIRST]
-  @State heightAdaptiveIndex: number = 0
+  fontSize:number = 16
+  showUnderline: boolean = false
+  maxLength: number = 6
+  letterSpacing: (number | string | Resource)[] = [-2, 0, 3, '5px', '10%']
+  letterSpacingIndex: number = 4
+  lineHeight: (number | string | Resource)[] = [-5, 0, 10, 20, '40vp', '30%']
+  lineHeightIndex: number = 0
+  heightTest: number = 80
+  widthTest: number = 250
+  paddingTest: number = 0
+  wordBreak: WordBreak[] = [WordBreak.NORMAL, WordBreak.BREAK_ALL, WordBreak.BREAK_WORD]
+  wordBreakIndex: number = 0
+  sizeAll: (number | string | Resource | undefined)[] = [-1, 0, 20, 'hsp', undefined, '20px', '10abc', $r('app.string.number_20px'), $r('app.string.number_10abc')]
+  maxFontSize: number = 2
+  minFontSize: number = 2
+  heightAdaptivePolicy: TextHeightAdaptivePolicy[] = [TextHeightAdaptivePolicy.MAX_LINES_FIRST, TextHeightAdaptivePolicy.MIN_FONT_SIZE_FIRST, TextHeightAdaptivePolicy.LAYOUT_CONSTRAINT_FIRST]
+  heightAdaptiveIndex: number = 0
   @State selectionStart: number = 0
   @State selectionEnd: number = 0
 
   build() {
     Column() {
       TextInput()
-      TextInput({ controller: this.controller, text: this.inputValue ,placeholder:'input your word...'})
```

## af85f1b4aa3d
- project: `TextComponentTest`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-redundant-state-var` (suggestion)
- file: `entry/src/main/ets/pages/32-Dev-2/Text.ets`:21:3
- patch: shape=`delete`, changed_lines=6

### Before (local context)
```
    15  
    16  @Entry
    17  @Component
    18  struct TextExample32dev {
    19    controller: TextController = new TextController();
    20    options: TextOptions = { controller: this.controller };
    21    @State text: string = 'Text editMenuOptions'
    22    @State phoneNumber: string = '13912345678';
    23    @State url: string = 'www.baidu.com';
    24    @State email: string = 'wangyi@example.com';
    25    @State address: string = '河北省保定市';
    26    @State datetime: string = '2024年7月19日';
    27    @State tag: boolean = true;
```

### After (local context)
```
    15  
    16  @Entry
    17  @Component
    18  struct TextExample32dev {
    19    controller: TextController = new TextController();
    20    options: TextOptions = { controller: this.controller };
    21    @State tag: boolean = true;
    22  
    23    onCreateMenu(menuItems: Array<TextMenuItem>) {
    24      menuItems.forEach((value, index) => {
    25        value.icon = $r('app.media.startIcon')
    26        if (value.id.equals(TextMenuItemId.COPY)) {
    27          value.content = $r('app.string.copy')
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/32-Dev-2/Text.ets
+++ after/entry/src/main/ets/pages/32-Dev-2/Text.ets
@@ -10,12 +10,6 @@
 struct TextExample32dev {
   controller: TextController = new TextController();
   options: TextOptions = { controller: this.controller };
-  @State text: string = 'Text editMenuOptions'
-  @State phoneNumber: string = '13912345678';
-  @State url: string = 'www.baidu.com';
-  @State email: string = 'wangyi@example.com';
-  @State address: string = '河北省保定市';
-  @State datetime: string = '2024年7月19日';
   @State tag: boolean = true;
 
   onCreateMenu(menuItems: Array<TextMenuItem>) {
@@ -23,3 +17,9 @@
       value.icon = $r('app.media.startIcon')
       if (value.id.equals(TextMenuItemId.COPY)) {
         value.content = $r('app.string.copy')
+      }
+      if (value.id.equals(TextMenuItemId.SELECT_ALL)) {
+        value.content = '全选change'
+      }
+    })
+    let item1: TextMenuItem = {
```

## a11894768140
- project: `TextComponentTest`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-redundant-state-var` (suggestion)
- file: `entry/src/main/ets/pages/50019/page10.ets`:23:3
- patch: shape=`modify`, changed_lines=11

### Before (local context)
```
    17  import { BusinessError } from '@ohos.base';
    18  import router from '@ohos.router'
    19  
    20  @Entry
    21  @Component
    22  struct SubWindowTextInput {
    23    @State stackA: NavPathStack = new NavPathStack();
    24    @State stackB: NavPathStack = new NavPathStack();
    25    private curWindow: window.Window | null = null;
    26    @State x: number = 50;
    27    @State y: number = 200;
    28    controller: TextInputController = new TextInputController()
    29    @State inputValue: string = ""
```

### After (local context)
```
    17  import window from '@ohos.window';
    18  import { BusinessError } from '@ohos.base';
    19  import router from '@ohos.router'
    20  
    21  @Entry
    22  @Component
    23  struct SubWindowTextInput {
    24    stackA: NavPathStack = new NavPathStack();
    25    stackB: NavPathStack = new NavPathStack();
    26    private curWindow: window.Window | null = null;
    27    // x 未发生变化，改为普通变量
    28    x: number = 50;
    29    // y 未发生变化，改为普通变量
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/50019/page10.ets
+++ after/entry/src/main/ets/pages/50019/page10.ets
@@ -1,3 +1,4 @@
+ * distributed under the License is distributed on an 'AS IS' BASIS,
  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  * See the License for the specific language governing permissions and
  * limitations under the License.
@@ -10,16 +11,15 @@
 @Entry
 @Component
 struct SubWindowTextInput {
-  @State stackA: NavPathStack = new NavPathStack();
-  @State stackB: NavPathStack = new NavPathStack();
+  stackA: NavPathStack = new NavPathStack();
+  stackB: NavPathStack = new NavPathStack();
   private curWindow: window.Window | null = null;
-  @State x: number = 50;
-  @State y: number = 200;
+  // x 未发生变化，改为普通变量
+  x: number = 50;
+  // y 未发生变化，改为普通变量
+  y: number = 200;
   controller: TextInputController = new TextInputController()
   @State inputValue: string = ""
-  @State isSupportAvoidance: boolean = true
-  @State marginTop: number = 0
-  @State cusHeight: number = 300
-  @State focusable1: boolean = true
-
-  // 自定义键盘组件
+  // isSupportAvoidance 未发生变化，改为普通变量
+  isSupportAvoidance: boolean = true
+  // marginTop 未发生变化，改为普通变量
```

## f0b18738f625
- project: `TextComponentTest`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-redundant-state-var` (suggestion)
- file: `entry/src/main/ets/pages/6-1/Text.ets`:20:3
- patch: shape=`delete`, changed_lines=2

### Before (local context)
```
    14   */
    15  
    16  @Entry
    17  @Component
    18  struct TextSpanExample {
    19    @State widthCon: number = 200
    20    @State heightCon: number = 60
    21    @State backgroundColorAll: (string | number | Color | Resource)[] = [Color.Blue, "#fc0303", 0xFF0000, 'rgb(2, 184, 17)', $r('app.color.color_yellow')]
    22    @State backgroundColorStr: string[] = ['Blue', "#fc0303", "0xFF0000", 'rgb(2, 184, 17)', "$r('yellow')"]
    23    @State backgroundColorIndex: number = 0
    24    @State radiusSpan: Dimension[] = ['10px', '20vp', '30fp', '50lpx', '10%', $r('app.string.radius')]
    25    @State radiusIndex: number = 0
    26  
```

### After (local context)
```
    14   */
    15  
    16  @Entry
    17  @Component
    18  struct TextSpanExample {
    19    @State backgroundColorAll: (string | number | Color | Resource)[] = [Color.Blue, "#fc0303", 0xFF0000, 'rgb(2, 184, 17)', $r('app.color.color_yellow')]
    20    @State backgroundColorStr: string[] = ['Blue', "#fc0303", "0xFF0000", 'rgb(2, 184, 17)', "$r('yellow')"]
    21    @State backgroundColorIndex: number = 0
    22    @State radiusSpan: Dimension[] = ['10px', '20vp', '30fp', '50lpx', '10%', $r('app.string.radius')]
    23    @State radiusIndex: number = 0
    24  
    25    build() {
    26      Column({ space: 8 }) {
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/6-1/Text.ets
+++ after/entry/src/main/ets/pages/6-1/Text.ets
@@ -9,8 +9,6 @@
 @Entry
 @Component
 struct TextSpanExample {
-  @State widthCon: number = 200
-  @State heightCon: number = 60
   @State backgroundColorAll: (string | number | Color | Resource)[] = [Color.Blue, "#fc0303", 0xFF0000, 'rgb(2, 184, 17)', $r('app.color.color_yellow')]
   @State backgroundColorStr: string[] = ['Blue', "#fc0303", "0xFF0000", 'rgb(2, 184, 17)', "$r('yellow')"]
   @State backgroundColorIndex: number = 0
@@ -23,3 +21,5 @@
         Span('Span').textBackgroundStyle({ color: this.backgroundColorAll[this.backgroundColorIndex], radius: this.radiusSpan[this.radiusIndex] }).borderWidth(1)
         ImageSpan($r('app.media.app_icon')).width(60).height(60)
           .textBackgroundStyle({ color: this.backgroundColorAll[this.backgroundColorIndex], radius: this.radiusSpan[this.radiusIndex] }).borderWidth(1)
+        ContainerSpan() {
+          Span('ContainerSpan').fontSize('16fp').fontColor(Color.White)
```

## 5c43ff672c6d
- project: `TextComponentTest`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-redundant-state-var` (suggestion)
- file: `entry/src/main/ets/pages/6-1/page9.ets`:24:3
- patch: shape=`modify`, changed_lines=3

### Before (local context)
```
    18  import window from '@ohos.window';
    19  import base from '@ohos.base';
    20  
    21  @Entry
    22  @Component
    23  struct MainWindow9800 {
    24    @State stackA: NavPathStack = new NavPathStack();
    25    @State stackB: NavPathStack = new NavPathStack();
    26    private myWindow: window.Window | null = null;
    27  
    28  
    29  
    30  
```

### After (local context)
```
    18  import observer from '@ohos.arkui.observer';
    19  import window from '@ohos.window';
    20  import base from '@ohos.base';
    21  
    22  @Entry
    23  @Component
    24  struct MainWindow9800 {
    25    private myWindow: window.Window | null = null;
    26  
    27  
    28  
    29  
    30    createWindow() {
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/6-1/page9.ets
+++ after/entry/src/main/ets/pages/6-1/page9.ets
@@ -1,3 +1,4 @@
+ * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  * See the License for the specific language governing permissions and
  * limitations under the License.
  */
@@ -10,8 +11,6 @@
 @Entry
 @Component
 struct MainWindow9800 {
-  @State stackA: NavPathStack = new NavPathStack();
-  @State stackB: NavPathStack = new NavPathStack();
   private myWindow: window.Window | null = null;
 
 
@@ -23,3 +22,4 @@
         .then((windowObj: window.Window) => {
           console.log(`testTag success to createWindow`)
           this.myWindow = windowObj;
+          this.myWindow.showWindow().then(() => {
```

## 4554436f1aba
- project: `ace_ets_component_common_attrss_flex001`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-container-without-property` (suggestion)
- file: `entry/src/main/ets/MainAbility/view/position/FlowItemView.ets`:40:11
- patch: shape=`modify`, changed_lines=7

### Before (local context)
```
    34          .height('100%')
    35          .position(this._position)
    36          .backgroundColor(Color.White)
    37          .key(this.componentKey)
    38  
    39          FlowItem() {
    40            Column() {
    41              Text('N2').fontSize(12).height('16')
    42            }
    43          }
    44          .width('100%')
    45          .height('100%')
    46          .backgroundColor(Color.White)
```

### After (local context)
```
    34          .width('100%')
    35          .height('100%')
    36          .position(this._position)
    37          .backgroundColor(Color.White)
    38          .key(this.componentKey)
    39  
    40          FlowItem() {
    41  
    42            Text('N2').fontSize(12).height('16')
    43            
    44          }
    45          .width('100%')
    46          .height('100%')
```

### Local diff
```diff
--- before/entry/src/main/ets/MainAbility/view/position/FlowItemView.ets
+++ after/entry/src/main/ets/MainAbility/view/position/FlowItemView.ets
@@ -1,7 +1,8 @@
+      WaterFlow() {
         FlowItem() {
-          Column() {
-            Text('N1').fontSize(12).height('16')
-          }
+
+          Text('N1').fontSize(12).height('16')
+          
         }
         .width('100%')
         .height('100%')
@@ -10,9 +11,9 @@
         .key(this.componentKey)
 
         FlowItem() {
-          Column() {
-            Text('N2').fontSize(12).height('16')
-          }
+
+          Text('N2').fontSize(12).height('16')
+          
         }
         .width('100%')
         .height('100%')
@@ -22,4 +23,3 @@
       .columnsTemplate('1fr 1fr')
       .backgroundColor(0xFAEEE0)
       .width(this.parentWidth)
-      .height(this.parentHeight)
```

## aa4ed163caa3
- project: `ace_ets_component_common_attrss_flex_nowear`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-container-without-property` (suggestion)
- file: `entry/src/main/ets/MainAbility/pages/flex/FlexBasisPage.ets`:111:11
- patch: shape=`modify`, changed_lines=16

### Before (local context)
```
   105        ScrollBar({ scroller: new Scroller(), state: BarState.On }) {
   106          Text().width(20).height(20).borderRadius(10)
   107        }.width(20).commonStyle()
   108      } else if (this.targetView == 'Stepper') {
   109        Stepper() {
   110          StepperItem() {
   111            Column() {
   112              Text('Page One')
   113            }
   114          }
   115        }.commonStyle()
   116      } else if (this.targetView == 'Search') {
   117        Search({ placeholder: 'Type to search...' })
```

### After (local context)
```
   105        ScrollBar({ scroller: new Scroller(), state: BarState.On }) {
   106          Text().width(20).height(20).borderRadius(10)
   107        }.width(20).commonStyle()
   108      } else if (this.targetView == 'Stepper') {
   109        Stepper() {
   110          StepperItem() {
   111  
   112            Text('Page One')
   113            
   114          }
   115        }.commonStyle()
   116      } else if (this.targetView == 'Search') {
   117        Search({ placeholder: 'Type to search...' })
```

### Local diff
```diff
--- before/entry/src/main/ets/MainAbility/pages/flex/FlexBasisPage.ets
+++ after/entry/src/main/ets/MainAbility/pages/flex/FlexBasisPage.ets
@@ -10,9 +10,9 @@
     } else if (this.targetView == 'Stepper') {
       Stepper() {
         StepperItem() {
-          Column() {
-            Text('Page One')
-          }
+
+          Text('Page One')
+          
         }
       }.commonStyle()
     } else if (this.targetView == 'Search') {
```

## 33c29bde7b65
- project: `ace_ets_component_common_attrss_flex_nowear`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-container-without-property` (suggestion)
- file: `entry/src/main/ets/MainAbility/view/position/FlowItemView.ets`:29:11
- patch: shape=`modify`, changed_lines=7

### Before (local context)
```
    23    private referenceComponentKey: string;
    24  
    25    build() {
    26      Column({ space: 2 }) {
    27        WaterFlow() {
    28          FlowItem() {
    29            Column() {
    30              Text('N1').fontSize(12).height('16')
    31            }
    32          }
    33          .width('100%')
    34          .height('100%')
    35          .position(this._position)
```

### After (local context)
```
    23    private parentComponentKey: string;
    24    private referenceComponentKey: string;
    25  
    26    build() {
    27      Column({ space: 2 }) {
    28        WaterFlow() {
    29          FlowItem() {
    30  
    31            Text('N1').fontSize(12).height('16')
    32            
    33          }
    34          .width('100%')
    35          .height('100%')
```

### Local diff
```diff
--- before/entry/src/main/ets/MainAbility/view/position/FlowItemView.ets
+++ after/entry/src/main/ets/MainAbility/view/position/FlowItemView.ets
@@ -1,3 +1,4 @@
+@Component
 export struct FlowItemView {
   @Link _position: Position;
   private componentKey: string;
@@ -10,9 +11,9 @@
     Column({ space: 2 }) {
       WaterFlow() {
         FlowItem() {
-          Column() {
-            Text('N1').fontSize(12).height('16')
-          }
+
+          Text('N1').fontSize(12).height('16')
+          
         }
         .width('100%')
         .height('100%')
@@ -21,5 +22,4 @@
         .key(this.componentKey)
 
         FlowItem() {
-          Column() {
-            Text('N2').fontSize(12).height('16')
+
```

## d4c4d749e795
- project: `ace_ets_component_common_attrss_flex_nowear`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-container-without-property` (suggestion)
- file: `entry/src/main/ets/MainAbility/view/position/WaterFlowView.ets`:46:5
- patch: shape=`modify`, changed_lines=16

### Before (local context)
```
    40    aboutToAppear() {
    41      this.getItemSizeArray()
    42    }
    43  
    44    @Builder
    45    itemFoot() {
    46      Column() {
    47        Text(`Footer`)
    48          .fontSize(10)
    49          .backgroundColor(Color.Red)
    50          .width(50)
    51          .height(50)
    52          .align(Alignment.Center)
```

### After (local context)
```
    40  
    41    aboutToAppear() {
    42      this.getItemSizeArray()
    43    }
    44  
    45    @Builder
    46    itemFoot() {
    47  
    48      Text(`Footer`)
    49        .fontSize(10)
    50        .backgroundColor(Color.Red)
    51        .width(50)
    52        .height(50)
```

### Local diff
```diff
--- before/entry/src/main/ets/MainAbility/view/position/WaterFlowView.ets
+++ after/entry/src/main/ets/MainAbility/view/position/WaterFlowView.ets
@@ -1,3 +1,4 @@
+  getItemSizeArray() {
     for (let i = 0; i < 100; i++) {
       this.itemWidthArray.push(this.getSize())
       this.itemHeightArray.push(this.getSize())
@@ -10,16 +11,15 @@
 
   @Builder
   itemFoot() {
-    Column() {
-      Text(`Footer`)
-        .fontSize(10)
-        .backgroundColor(Color.Red)
-        .width(50)
-        .height(50)
-        .align(Alignment.Center)
-        .margin({ top: 2 })
-    }
+
+    Text(`Footer`)
+      .fontSize(10)
+      .backgroundColor(Color.Red)
+      .width(50)
+      .height(50)
+      .align(Alignment.Center)
+      .margin({ top: 2 })
+    
   }
 
   build() {
-    WaterFlow({ footer: this.itemFoot.bind(this), scroller: this.scroller }) {
```

## 2a566887c5cf
- project: `ace_ets_module_imageText_api12`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-container-without-property` (suggestion)
- file: `entry/src/main/ets/MainAbility/pages/RichEditorEnhance01/RichEditor_TextStyle.ets`:119:11
- patch: shape=`modify`, changed_lines=10

### Before (local context)
```
   113              Text("getSpans获取到的字符间距为：")
   114              Text(this.content_L).id("StyleLetterSpacing_" + this.content_L).fontSize(15).fontColor(Color.Green)
   115            }
   116            Row(){
   117              Text("LineHeight:" + this.LH).width("100%")
   118            }
   119            Row(){
   120              Text("LetterSpacing:" + this.LS).width("100%")
   121            }
   122          }
   123        }
   124        .borderWidth(1)
   125        .borderColor(Color.Red)
```

### After (local context)
```
   113            }
   114            Row(){
   115              Text("getSpans获取到的字符间距为：")
   116              Text(this.content_L).id("StyleLetterSpacing_" + this.content_L).fontSize(15).fontColor(Color.Green)
   117            }
   118  
   119            Text("LineHeight:" + this.LH).width("100%")
   120            
   121  
   122            Text("LetterSpacing:" + this.LS).width("100%")
   123            
   124          }
   125        }
```

### Local diff
```diff
--- before/entry/src/main/ets/MainAbility/pages/RichEditorEnhance01/RichEditor_TextStyle.ets
+++ after/entry/src/main/ets/MainAbility/pages/RichEditorEnhance01/RichEditor_TextStyle.ets
@@ -1,3 +1,5 @@
+
+      Scroll(){
         Column({space: 10}) {
           Row(){
             Text("getSpans获取到的行高为：")
@@ -7,12 +9,12 @@
             Text("getSpans获取到的字符间距为：")
             Text(this.content_L).id("StyleLetterSpacing_" + this.content_L).fontSize(15).fontColor(Color.Green)
           }
-          Row(){
-            Text("LineHeight:" + this.LH).width("100%")
-          }
-          Row(){
-            Text("LetterSpacing:" + this.LS).width("100%")
-          }
+
+          Text("LineHeight:" + this.LH).width("100%")
+          
+
+          Text("LetterSpacing:" + this.LS).width("100%")
+          
         }
       }
       .borderWidth(1)
@@ -21,5 +23,3 @@
       .height("20%")
       .margin({bottom: 20})
 
-      Column() {
-        RichEditor(this.options).clip(true).padding(10)
```

## 59f5c2ae54a4
- project: `ace_ets_module_imageText_api12`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-unchanged-state-var` (suggestion)
- file: `entry/src/main/ets/MainAbility/pages/RichEditorStringSpan/RichEditorStringSpan003.ets`:28:3
- patch: shape=`delete`, changed_lines=5

### Before (local context)
```
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
    34      fontSize: LengthMetrics.vp(30),
```

### After (local context)
```
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
    34      fontWeight: FontWeight.Bolder,
```

### Local diff
```diff
--- before/entry/src/main/ets/MainAbility/pages/RichEditorStringSpan/RichEditorStringSpan003.ets
+++ after/entry/src/main/ets/MainAbility/pages/RichEditorStringSpan/RichEditorStringSpan003.ets
@@ -5,12 +5,7 @@
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
   fontStyle2: TextStyle = new TextStyle({
     fontWeight: FontWeight.Bolder,
     fontFamily: 'Arial',
+    fontColor: Color.Orange,
+    fontSize: LengthMetrics.vp(50),
+    fontStyle: FontStyle.Italic
+  })
+  // 创建属性字符串对象
```

## d7d24aff3460
- project: `ace_ets_module_imageText_api12`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-unchanged-state-var` (suggestion)
- file: `entry/src/main/ets/MainAbility/pages/TextArea/TextAreaApi2.ets`:18:3
- patch: shape=`delete`, changed_lines=1

### Before (local context)
```
    12   * See the License for the specific language governing permissions and
    13   * limitations under the License.
    14   */
    15  @Entry
    16  @Component
    17  struct TextAreaApi2 {
    18    @State text1: string = ''
    19  
    20    build() {
    21      Column() {
    22        TextArea({ placeholder: 'input your email...' })
    23          .width('95%')
    24          .height(40)
```

### After (local context)
```
    12   * See the License for the specific language governing permissions and
    13   * limitations under the License.
    14   */
    15  @Entry
    16  @Component
    17  struct TextAreaApi2 {
    18  
    19    build() {
    20      Column() {
    21        TextArea({ placeholder: 'input your email...' })
    22          .width('95%')
    23          .height(40)
    24          .margin(20)
```

### Local diff
```diff
--- before/entry/src/main/ets/MainAbility/pages/TextArea/TextAreaApi2.ets
+++ after/entry/src/main/ets/MainAbility/pages/TextArea/TextAreaApi2.ets
@@ -10,7 +10,6 @@
 @Entry
 @Component
 struct TextAreaApi2 {
-  @State text1: string = ''
 
   build() {
     Column() {
@@ -23,3 +22,4 @@
         .maxLength(20)
         .id('TextAreaApi2_textArea')
 
+
```

## fec4fce4e166
- project: `ace_ets_module_imageText_api16`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-unchanged-state-var` (suggestion)
- file: `entry/src/main/ets/MainAbility/pages/HalfLeadingTest/searchHalfLeadingTest.ets`:19:3
- patch: shape=`modify`, changed_lines=25

### Before (local context)
```
    13   * limitations under the License.
    14   */
    15  
    16  @Entry
    17  @Component
    18  struct searchHalfLeadingTest {
    19    @State text: string = 'As the sun begins to set, casting a warm golden hue across the sky'
    20    build() {
    21      Row() {
    22        Column({ space: 20 }) {
    23          Search({ value:this.text})
    24            .halfLeading(true)
    25            .id('search1')
```

### After (local context)
```
    13   * See the License for the specific language governing permissions and
    14   * limitations under the License.
    15   */
    16  
    17  @Entry
    18  @Component
    19  struct searchHalfLeadingTest {
    20    // 直接使用一般变量即可
    21    text: string = 'As the sun begins to set, casting a warm golden hue across the sky'
    22    build() {
    23  
    24      Column({ space: 20 }) {
    25        Search({ value:this.text})
```

### Local diff
```diff
--- before/entry/src/main/ets/MainAbility/pages/HalfLeadingTest/searchHalfLeadingTest.ets
+++ after/entry/src/main/ets/MainAbility/pages/HalfLeadingTest/searchHalfLeadingTest.ets
@@ -1,3 +1,4 @@
+ *
  *     http://www.apache.org/licenses/LICENSE-2.0
  *
  * Unless required by applicable law or agreed to in writing, software
@@ -10,16 +11,15 @@
 @Entry
 @Component
 struct searchHalfLeadingTest {
-  @State text: string = 'As the sun begins to set, casting a warm golden hue across the sky'
+  // 直接使用一般变量即可
+  text: string = 'As the sun begins to set, casting a warm golden hue across the sky'
   build() {
-    Row() {
-      Column({ space: 20 }) {
-        Search({ value:this.text})
-          .halfLeading(true)
-          .id('search1')
-          .height('10%')
-        Search({ value: this.text})
-          .halfLeading(false)
-          .id('search2')
-          .height('10%')
-        Search({ value: this.text})
+
+    Column({ space: 20 }) {
+      Search({ value:this.text})
+        .halfLeading(true)
+        .id('search1')
+        .height('10%')
+      Search({ value: this.text})
+        .halfLeading(false)
+        .id('search2')
```

## 770226bdc5e5
- project: `ace_ets_module_imageText_api16`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-redundant-state-var` (suggestion)
- file: `entry/src/main/ets/MainAbility/pages/Hyperlink/preventDefault.ets`:25:3
- patch: shape=`delete`, changed_lines=10

### Before (local context)
```
    19  struct Index {
    20    @State color:Color = Color.Pink
    21    @State isShow: boolean = false
    22    @State isShow2: boolean = false
    23    @State sheetHeight: number = 300;
    24    @State isShowdialog: boolean = false
    25    @State handlePopup: boolean = false
    26    @State customPopup: boolean = false
    27    @State text1: string = ''
    28    @State text2: string = ''
    29    controller1: TextController = new TextController();
    30    styledString2: StyledString = new StyledString('运动45分钟');
    31    mutableStyledString2: MutableStyledString = new MutableStyledString('test hello world', [{
```

### After (local context)
```
    19  struct Index {
    20    @State color:Color = Color.Pink
    21    controller1: TextController = new TextController();
    22    styledString2: StyledString = new StyledString('运动45分钟');
    23    mutableStyledString2: MutableStyledString = new MutableStyledString('test hello world', [{
    24      start: 0,
    25      length: 5,
    26      styledKey: StyledStringKey.FONT,
    27      styledValue: new TextStyle({ fontColor: Color.Blue })
    28    }]);
    29  
    30    searchController: SearchController = new SearchController()
    31  
```

### Local diff
```diff
--- before/entry/src/main/ets/MainAbility/pages/Hyperlink/preventDefault.ets
+++ after/entry/src/main/ets/MainAbility/pages/Hyperlink/preventDefault.ets
@@ -6,14 +6,6 @@
 @Component
 struct Index {
   @State color:Color = Color.Pink
-  @State isShow: boolean = false
-  @State isShow2: boolean = false
-  @State sheetHeight: number = 300;
-  @State isShowdialog: boolean = false
-  @State handlePopup: boolean = false
-  @State customPopup: boolean = false
-  @State text1: string = ''
-  @State text2: string = ''
   controller1: TextController = new TextController();
   styledString2: StyledString = new StyledString('运动45分钟');
   mutableStyledString2: MutableStyledString = new MutableStyledString('test hello world', [{
@@ -23,3 +15,11 @@
     styledValue: new TextStyle({ fontColor: Color.Blue })
   }]);
 
+  searchController: SearchController = new SearchController()
+
+  async onPageShow() {
+    this.controller1.setStyledString(this.mutableStyledString2)
+  }
+
+  build() {
+    Column({space:20}) {
```

## a01c56051281
- project: `ace_ets_module_imageText_api16`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-unchanged-state-var` (suggestion)
- file: `entry/src/main/ets/MainAbility/pages/Hyperlink/preventDefault.ets`:27:3
- patch: shape=`delete`, changed_lines=10

### Before (local context)
```
    21    @State isShow: boolean = false
    22    @State isShow2: boolean = false
    23    @State sheetHeight: number = 300;
    24    @State isShowdialog: boolean = false
    25    @State handlePopup: boolean = false
    26    @State customPopup: boolean = false
    27    @State text1: string = ''
    28    @State text2: string = ''
    29    controller1: TextController = new TextController();
    30    styledString2: StyledString = new StyledString('运动45分钟');
    31    mutableStyledString2: MutableStyledString = new MutableStyledString('test hello world', [{
    32      start: 0,
    33      length: 5,
```

### After (local context)
```
    21    controller1: TextController = new TextController();
    22    styledString2: StyledString = new StyledString('运动45分钟');
    23    mutableStyledString2: MutableStyledString = new MutableStyledString('test hello world', [{
    24      start: 0,
    25      length: 5,
    26      styledKey: StyledStringKey.FONT,
    27      styledValue: new TextStyle({ fontColor: Color.Blue })
    28    }]);
    29  
    30    searchController: SearchController = new SearchController()
    31  
    32    async onPageShow() {
    33      this.controller1.setStyledString(this.mutableStyledString2)
```

### Local diff
```diff
--- before/entry/src/main/ets/MainAbility/pages/Hyperlink/preventDefault.ets
+++ after/entry/src/main/ets/MainAbility/pages/Hyperlink/preventDefault.ets
@@ -4,14 +4,6 @@
 @Component
 struct Index {
   @State color:Color = Color.Pink
-  @State isShow: boolean = false
-  @State isShow2: boolean = false
-  @State sheetHeight: number = 300;
-  @State isShowdialog: boolean = false
-  @State handlePopup: boolean = false
-  @State customPopup: boolean = false
-  @State text1: string = ''
-  @State text2: string = ''
   controller1: TextController = new TextController();
   styledString2: StyledString = new StyledString('运动45分钟');
   mutableStyledString2: MutableStyledString = new MutableStyledString('test hello world', [{
@@ -22,4 +14,12 @@
   }]);
 
   searchController: SearchController = new SearchController()
-  @State changeValue: string = ''
+
+  async onPageShow() {
+    this.controller1.setStyledString(this.mutableStyledString2)
+  }
+
+  build() {
+    Column({space:20}) {
+      Column() {
+        Text('禁掉onclick')
```

## 0b65fc5012e1
- project: `ace_ets_module_navigation1`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-unchanged-state-var` (suggestion)
- file: `entry/src/main/ets/MainAbility/pages/Navigation/RouterRecoverable.ets`:22:3
- patch: shape=`modify`, changed_lines=4

### Before (local context)
```
    16      this.data2 = new InnerParams(tuple)
    17    }
    18  }
    19  @Entry
    20  @Component
    21  struct RouterRecoverable {
    22    @State message: string = 'routerPage1';
    23  
    24    @State text1:   string = '这里显示栈信息'
    25    @State text2:   number|string = 0
    26    controller: TextInputController = new TextInputController()
    27    routerPageUpdateCallback(info: RouterPageInfo) {
    28      if(info){
```

### After (local context)
```
    16      this.data1 = str
    17      this.data2 = new InnerParams(tuple)
    18    }
    19  }
    20  @Entry
    21  @Component
    22  struct RouterRecoverable {
    23  
    24    controller: TextInputController = new TextInputController()
    25    routerPageUpdateCallback(info: RouterPageInfo) {
    26      if(info){
    27         console.log(`testTag routerPageUpdateCallback, index: ${info.index}, name: ${info.name}, path: ${info.path}, state: ${info.state}, pageId: ${info.pageId}`);
    28      }
```

### Local diff
```diff
--- before/entry/src/main/ets/MainAbility/pages/Navigation/RouterRecoverable.ets
+++ after/entry/src/main/ets/MainAbility/pages/Navigation/RouterRecoverable.ets
@@ -1,3 +1,4 @@
+}
 class RouterParams {
   private data1: string
   private data2: InnerParams
@@ -10,10 +11,7 @@
 @Entry
 @Component
 struct RouterRecoverable {
-  @State message: string = 'routerPage1';
 
-  @State text1:   string = '这里显示栈信息'
-  @State text2:   number|string = 0
   controller: TextInputController = new TextInputController()
   routerPageUpdateCallback(info: RouterPageInfo) {
     if(info){
@@ -23,3 +21,5 @@
 
   build() {
     Column({ space: '20vp' }) {
+      Button('routerPage2_recoverable=true', { stateEffect: true, type: ButtonType.Capsule })
+        .id('routerBtnId1')
```

## dd13ec09525b
- project: `ace_ets_module_navigation1`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-unchanged-state-var` (suggestion)
- file: `entry/src/main/ets/MainAbility/pages/Navigation/navition3.ets`:20:3
- patch: shape=`delete`, changed_lines=10

### Before (local context)
```
    14   */
    15  @Entry
    16  @Component
    17  struct navigationType {
    18    @State strokeWidthStr: string = '10px'
    19    @State dividerColorStr: string  = '#000000'
    20    @State startMarginStr: string = '5px'
    21    @State endMarginStr: string = '5px'
    22  
    23    @State nullFlag: boolean = false
    24  
    25    @State dividerColor: Color = Color.Red
    26  
```

### After (local context)
```
    14   */
    15  @Entry
    16  @Component
    17  struct navigationType {
    18    private arr: number[] = [1, 2, 3];
    19    normalIcon : Resource = $r("app.media.icon")
    20    selectedIcon: Resource = $r("app.media.icon")
    21  
    22    build() {
    23      Column() {
    24        Navigation() {
    25          TextInput({ placeholder: 'search...' })
    26            .width("90%")
```

### Local diff
```diff
--- before/entry/src/main/ets/MainAbility/pages/Navigation/navition3.ets
+++ after/entry/src/main/ets/MainAbility/pages/Navigation/navition3.ets
@@ -8,18 +8,18 @@
 @Entry
 @Component
 struct navigationType {
-  @State strokeWidthStr: string = '10px'
-  @State dividerColorStr: string  = '#000000'
-  @State startMarginStr: string = '5px'
-  @State endMarginStr: string = '5px'
-
-  @State nullFlag: boolean = false
-
-  @State dividerColor: Color = Color.Red
-
-
   private arr: number[] = [1, 2, 3];
   normalIcon : Resource = $r("app.media.icon")
   selectedIcon: Resource = $r("app.media.icon")
 
   build() {
+    Column() {
+      Navigation() {
+        TextInput({ placeholder: 'search...' })
+          .width("90%")
+          .height(40)
+          .backgroundColor('#FFFFFF')
+
+        List({ space: 12 }) {
+          ForEach(this.arr, (item: number) => {
+            ListItem() {
```

## 158574db38f7
- project: `ace_ets_module_navigation1`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-redundant-state-var` (suggestion)
- file: `entry/src/main/ets/MainAbility/pages/Navigation/navition3.ets`:20:3
- patch: shape=`delete`, changed_lines=10

### Before (local context)
```
    14   */
    15  @Entry
    16  @Component
    17  struct navigationType {
    18    @State strokeWidthStr: string = '10px'
    19    @State dividerColorStr: string  = '#000000'
    20    @State startMarginStr: string = '5px'
    21    @State endMarginStr: string = '5px'
    22  
    23    @State nullFlag: boolean = false
    24  
    25    @State dividerColor: Color = Color.Red
    26  
```

### After (local context)
```
    14   */
    15  @Entry
    16  @Component
    17  struct navigationType {
    18    private arr: number[] = [1, 2, 3];
    19    normalIcon : Resource = $r("app.media.icon")
    20    selectedIcon: Resource = $r("app.media.icon")
    21  
    22    build() {
    23      Column() {
    24        Navigation() {
    25          TextInput({ placeholder: 'search...' })
    26            .width("90%")
```

### Local diff
```diff
--- before/entry/src/main/ets/MainAbility/pages/Navigation/navition3.ets
+++ after/entry/src/main/ets/MainAbility/pages/Navigation/navition3.ets
@@ -8,18 +8,18 @@
 @Entry
 @Component
 struct navigationType {
-  @State strokeWidthStr: string = '10px'
-  @State dividerColorStr: string  = '#000000'
-  @State startMarginStr: string = '5px'
-  @State endMarginStr: string = '5px'
-
-  @State nullFlag: boolean = false
-
-  @State dividerColor: Color = Color.Red
-
-
   private arr: number[] = [1, 2, 3];
   normalIcon : Resource = $r("app.media.icon")
   selectedIcon: Resource = $r("app.media.icon")
 
   build() {
+    Column() {
+      Navigation() {
+        TextInput({ placeholder: 'search...' })
+          .width("90%")
+          .height(40)
+          .backgroundColor('#FFFFFF')
+
+        List({ space: 12 }) {
+          ForEach(this.arr, (item: number) => {
+            ListItem() {
```

## 9ddf34d6d46b
- project: `ace_ets_module_navigation1`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-avoid-empty-callback` (suggestion)
- file: `entry/src/main/ets/MainAbility/pages/index/index.ets`:46:7
- patch: shape=`add`, changed_lines=2

### Before (local context)
```
    40            .fontWeight(FontWeight.Bold)
    41        }.type(ButtonType.Capsule)
    42        .margin({
    43          top: 20
    44        })
    45        .backgroundColor('#0D9FFB')
    46        .onClick(() => {
    47        })
    48      }
    49      .width('100%')
    50      .height('100%')
    51    }
    52  }
```

### After (local context)
```
    40            .fontSize(25)
    41            .fontWeight(FontWeight.Bold)
    42        }.type(ButtonType.Capsule)
    43        .margin({
    44          top: 20
    45        })
    46        .backgroundColor('#0D9FFB')
    47        .onClick(() => {
    48          this.handleScrollEvent()
    49        })
    50      }
    51      .width('100%')
    52      .height('100%')
```

### Local diff
```diff
--- before/entry/src/main/ets/MainAbility/pages/index/index.ets
+++ after/entry/src/main/ets/MainAbility/pages/index/index.ets
@@ -1,3 +1,4 @@
+    Flex({ direction:FlexDirection.Column, alignItems:ItemAlign.Center, justifyContent: FlexAlign.Center }) {
       Text('Hello World')
         .fontSize(50)
         .fontWeight(FontWeight.Bold)
@@ -11,6 +12,7 @@
       })
       .backgroundColor('#0D9FFB')
       .onClick(() => {
+        this.handleScrollEvent()
       })
     }
     .width('100%')
```

## 71b890027d59
- project: `ace_ets_module_navigation_api12`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-redundant-state-var` (suggestion)
- file: `entry/src/main/ets/MainAbility/pages/MultiNavigation/MultiNavigation29.ets`:24:3
- patch: shape=`modify`, changed_lines=22

### Before (local context)
```
    18  @Entry
    19  @Component
    20  struct MultiNavigation29 {
    21    @Provide('pageStack') pageStack: MultiNavPathStack = new MultiNavPathStack();
    22    @State text:string = ''
    23    @State text2:string = ''
    24    @State text3:string = ''
    25  
    26    aboutToAppear(): void {
    27      this.pageStack.pushPath({
    28        name: 'PageHome1',
    29        param: "PageHome1",
    30        onPop: (popInfo) => {
```

### After (local context)
```
    18  
    19  @Entry
    20  @Component
    21  struct MultiNavigation29 {
    22    @Provide('pageStack') pageStack: MultiNavPathStack = new MultiNavPathStack();
    23    @State text:string = ''
    24  
    25    aboutToAppear(): void {
    26      this.pageStack.pushPath({
    27        name: 'PageHome1',
    28        param: "PageHome1",
    29        onPop: (popInfo) => {
    30        }
```

### Local diff
```diff
--- before/entry/src/main/ets/MainAbility/pages/MultiNavigation/MultiNavigation29.ets
+++ after/entry/src/main/ets/MainAbility/pages/MultiNavigation/MultiNavigation29.ets
@@ -1,3 +1,4 @@
+ * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  * See the License for the specific language governing permissions and
  * limitations under the License.
  */
@@ -9,8 +10,6 @@
 struct MultiNavigation29 {
   @Provide('pageStack') pageStack: MultiNavPathStack = new MultiNavPathStack();
   @State text:string = ''
-  @State text2:string = ''
-  @State text3:string = ''
 
   aboutToAppear(): void {
     this.pageStack.pushPath({
@@ -23,3 +22,4 @@
   }
 
   @Builder
+  PageMap(name: string, param?: object) {
```

## 2b8eb56a980f
- project: `ace_ets_module_navigation_api12`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-container-without-property` (suggestion)
- file: `entry/src/main/ets/MainAbility/pages/Navigation/NavPathStack1_6.ets`:151:7
- patch: shape=`modify`, changed_lines=10

### Before (local context)
```
   145  @Component
   146  struct  PageThree{
   147    derivedStack:DerivedNavPathStack = new DerivedNavPathStack()
   148  
   149    build() {
   150      NavDestination(){
   151        Column(){
   152          Text('pageThree').id('NavPathStack1_6_pageThree_text1')
   153        }
   154      }
   155      .title('pageThree')
   156    }
   157  }
```

### After (local context)
```
   145  
   146  @Component
   147  struct  PageThree{
   148    derivedStack:DerivedNavPathStack = new DerivedNavPathStack()
   149  
   150    build() {
   151      NavDestination(){
   152  
   153        Text('pageThree').id('NavPathStack1_6_pageThree_text1')
   154        
   155      }
   156      .title('pageThree')
   157    }
```

### Local diff
```diff
--- before/entry/src/main/ets/MainAbility/pages/Navigation/NavPathStack1_6.ets
+++ after/entry/src/main/ets/MainAbility/pages/Navigation/NavPathStack1_6.ets
@@ -1,4 +1,5 @@
-      }
+      Text('pageTwo').id('NavPathStack1_6_pageTwo_text1')
+      
     }
     .title('pageTwo')
   }
@@ -10,9 +11,9 @@
 
   build() {
     NavDestination(){
-      Column(){
-        Text('pageThree').id('NavPathStack1_6_pageThree_text1')
-      }
+
+      Text('pageThree').id('NavPathStack1_6_pageThree_text1')
+      
     }
     .title('pageThree')
   }
```

## b8d621361d10
- project: `ace_ets_module_navigation_api12`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-container-without-property` (suggestion)
- file: `entry/src/main/ets/MainAbility/pages/Navigation/NavPathStack2.ets`:203:7
- patch: shape=`modify`, changed_lines=10

### Before (local context)
```
   197  @Component
   198  struct  PageThree{
   199    derivedStack:DerivedNavPathStack = new DerivedNavPathStack()
   200  
   201    build() {
   202      NavDestination(){
   203        Column(){
   204          Text('pageThree').id('NavPathStack2_pageThree_text1')
   205        }
   206      }
   207      .title('pageThree')
   208    }
   209  }
```

### After (local context)
```
   197  
   198  @Component
   199  struct  PageThree{
   200    derivedStack:DerivedNavPathStack = new DerivedNavPathStack()
   201  
   202    build() {
   203      NavDestination(){
   204  
   205        Text('pageThree').id('NavPathStack2_pageThree_text1')
   206        
   207      }
   208      .title('pageThree')
   209    }
```

### Local diff
```diff
--- before/entry/src/main/ets/MainAbility/pages/Navigation/NavPathStack2.ets
+++ after/entry/src/main/ets/MainAbility/pages/Navigation/NavPathStack2.ets
@@ -1,4 +1,5 @@
-      }
+      Text('pageTwo').id('NavPathStack2_pageTwo_text1')
+      
     }
     .title('pageTwo')
   }
@@ -10,9 +11,9 @@
 
   build() {
     NavDestination(){
-      Column(){
-        Text('pageThree').id('NavPathStack2_pageThree_text1')
-      }
+
+      Text('pageThree').id('NavPathStack2_pageThree_text1')
+      
     }
     .title('pageThree')
   }
```

## f1eb2acdf2d5
- project: `ace_ets_module_navigation_api12`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-container-without-property` (suggestion)
- file: `entry/src/main/ets/MainAbility/pages/Navigation/NavPathStack2_4.ets`:204:7
- patch: shape=`modify`, changed_lines=10

### Before (local context)
```
   198  @Component
   199  struct  PageThree{
   200    derivedStack:DerivedNavPathStack = new DerivedNavPathStack()
   201  
   202    build() {
   203      NavDestination(){
   204        Column(){
   205          Text('pageThree').id('NavPathStack2_4_pageThree_text1')
   206        }
   207      }
   208      .title('pageThree')
   209    }
   210  }
```

### After (local context)
```
   198  
   199  @Component
   200  struct  PageThree{
   201    derivedStack:DerivedNavPathStack = new DerivedNavPathStack()
   202  
   203    build() {
   204      NavDestination(){
   205  
   206        Text('pageThree').id('NavPathStack2_4_pageThree_text1')
   207        
   208      }
   209      .title('pageThree')
   210    }
```

### Local diff
```diff
--- before/entry/src/main/ets/MainAbility/pages/Navigation/NavPathStack2_4.ets
+++ after/entry/src/main/ets/MainAbility/pages/Navigation/NavPathStack2_4.ets
@@ -1,4 +1,5 @@
-      }
+      Text('pageTwo').id('NavPathStack2_4_pageTwo_text1')
+      
     }
     .title('pageTwo')
   }
@@ -10,9 +11,9 @@
 
   build() {
     NavDestination(){
-      Column(){
-        Text('pageThree').id('NavPathStack2_4_pageThree_text1')
-      }
+
+      Text('pageThree').id('NavPathStack2_4_pageThree_text1')
+      
     }
     .title('pageThree')
   }
```

## d2d2b7a06947
- project: `ace_ets_module_navigation_api12`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-unchanged-state-var` (suggestion)
- file: `entry/src/main/ets/MainAbility/pages/Navigation/NavigationLevel/NavigationLevelTest20.ets`:19:3
- patch: shape=`delete`, changed_lines=2

### Before (local context)
```
    13   * limitations under the License.
    14   */
    15  
    16  @Entry
    17  @Component
    18  struct NavigationLevelTest20 {
    19    @State message:string = 'NavBar'
    20    @State pageInfos: NavPathStack = new NavPathStack();
    21  
    22    @Builder
    23    pageOneTmp() {
    24      NavDestination() {
    25        Column({ space: 10 }) {
```

### After (local context)
```
    13   * limitations under the License.
    14   */
    15  
    16  @Entry
    17  @Component
    18  struct NavigationLevelTest20 {
    19    @State pageInfos: NavPathStack = new NavPathStack();
    20  
    21    @Builder
    22    pageOneTmp() {
    23      NavDestination() {
    24        Column({ space: 10 }) {
    25          Text('pageOne').id('NavigationLevelTest20_text1')
```

### Local diff
```diff
--- before/entry/src/main/ets/MainAbility/pages/Navigation/NavigationLevel/NavigationLevelTest20.ets
+++ after/entry/src/main/ets/MainAbility/pages/Navigation/NavigationLevel/NavigationLevelTest20.ets
@@ -10,7 +10,6 @@
 @Entry
 @Component
 struct NavigationLevelTest20 {
-  @State message:string = 'NavBar'
   @State pageInfos: NavPathStack = new NavPathStack();
 
   @Builder
@@ -23,3 +22,4 @@
   }
 
   @Builder
+  pageTwoTmp() {
```

## 2ddd02c9bc81
- project: `ace_ets_module_navigation_api12`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-container-without-property` (suggestion)
- file: `entry/src/main/ets/MainAbility/pages/Navigation/Navigation_SingleInstanceJump/Navigation_SingleInstanceJump27.ets`:37:7
- patch: shape=`modify`, changed_lines=11

### Before (local context)
```
    31      }
    32    }
    33  
    34  
    35    @Builder Page1(){
    36      NavDestination(){
    37        Column(){
    38          Text('page1')
    39        }
    40      }
    41      .title('page1')
    42  
    43    }
```

### After (local context)
```
    31         text += `${name[i]},Param:${JSON.stringify(this.pathInfos.getParamByIndex(i))} `
    32        }
    33      }
    34     this.text = text
    35    }
    36  
    37  
    38    @Builder Page1(){
    39      NavDestination(){
    40  
    41        Text('page1')
    42        
    43      }
```

### Local diff
```diff
--- before/entry/src/main/ets/MainAbility/pages/Navigation/Navigation_SingleInstanceJump/Navigation_SingleInstanceJump27.ets
+++ after/entry/src/main/ets/MainAbility/pages/Navigation/Navigation_SingleInstanceJump/Navigation_SingleInstanceJump27.ets
@@ -1,25 +1,25 @@
+    let name:string[] = this.pathInfos.getAllPathName()
+   let text = this.text
     for (let i = 0; i < name.length; i++) {
       if (!this.pathInfos.getParamByIndex(i)) {
-        this.text += ` ${name[i]},Param:undefined`
+       text += ` ${name[i]},Param:undefined`
       }else{
-        this.text += `${name[i]},Param:${JSON.stringify(this.pathInfos.getParamByIndex(i))} `
+       text += `${name[i]},Param:${JSON.stringify(this.pathInfos.getParamByIndex(i))} `
       }
     }
+   this.text = text
   }
 
 
   @Builder Page1(){
     NavDestination(){
-      Column(){
-        Text('page1')
-      }
+
+      Text('page1')
+      
     }
     .title('page1')
 
   }
   @Builder Page2(){
     NavDestination(){
-      Column(){
-        Text('page1')
-      }
-    }.title('page1')
+
```

## f6e9956bdc6f
- project: `ace_ets_module_nowear_waterflow`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-unchanged-state-var` (suggestion)
- file: `entry/src/main/ets/MainAbility/pages/ScrollToIndex/ScrollToIndexForWaterFlow02.ets`:22:3
- patch: shape=`modify`, changed_lines=15

### Before (local context)
```
    16  
    17  @Entry
    18  @Component
    19  struct ScrollToIndexForWaterFlow02 {
    20    @State minSize: number = 80
    21    @State maxSize: number = 180
    22    @State fontSize: number = 24
    23    @State colors: number[] = [0xFFC0CB, 0xDA70D6, 0x6B8E23, 0x6A5ACD, 0x00FFFF, 0x00FF7F]
    24    private scrollerForWaterFlow: Scroller = new Scroller()
    25    private itemScrollerForWaterFlow: Scroller = new Scroller()
    26    datasource: WaterFlowDataSource = new WaterFlowDataSource()
    27    private itemWidthArray: number[] = []
    28    private itemHeightArray: number[] = []
```

### After (local context)
```
    16  import { WaterFlowDataSource } from '../WaterFlowDataSource'
    17  
    18  @Entry
    19  @Component
    20  struct ScrollToIndexForWaterFlow02 {
    21    minSize: number = 80
    22    maxSize: number = 180
    23    fontSize: number = 24
    24    @State colors: number[] = [0xFFC0CB, 0xDA70D6, 0x6B8E23, 0x6A5ACD, 0x00FFFF, 0x00FF7F]
    25    private scrollerForWaterFlow: Scroller = new Scroller()
    26    private itemScrollerForWaterFlow: Scroller = new Scroller()
    27    datasource: WaterFlowDataSource = new WaterFlowDataSource()
    28    private itemWidthArray: number[] = []
```

### Local diff
```diff
--- before/entry/src/main/ets/MainAbility/pages/ScrollToIndex/ScrollToIndexForWaterFlow02.ets
+++ after/entry/src/main/ets/MainAbility/pages/ScrollToIndex/ScrollToIndexForWaterFlow02.ets
@@ -1,3 +1,4 @@
+ * Unless required by applicable law or agreed to in writing, software
  * distributed under the License is distributed on an 'AS IS' BASIS,
  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  * See the License for the specific language governing permissions and
@@ -8,9 +9,9 @@
 @Entry
 @Component
 struct ScrollToIndexForWaterFlow02 {
-  @State minSize: number = 80
-  @State maxSize: number = 180
-  @State fontSize: number = 24
+  minSize: number = 80
+  maxSize: number = 180
+  fontSize: number = 24
   @State colors: number[] = [0xFFC0CB, 0xDA70D6, 0x6B8E23, 0x6A5ACD, 0x00FFFF, 0x00FF7F]
   private scrollerForWaterFlow: Scroller = new Scroller()
   private itemScrollerForWaterFlow: Scroller = new Scroller()
@@ -22,4 +23,3 @@
   // 计算flow item宽/高
   getSize() {
     let ret = Math.floor(Math.random() * this.maxSize)
-    return (ret > this.minSize ? ret : this.minSize)
```

## 64433b3044a9
- project: `ace_ets_module_nowear_waterflow`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-set-cache-count-for-lazyforeach-grid` (suggestion)
- file: `entry/src/main/ets/MainAbility/pages/ScrollToIndex/ScrollToIndexForWaterFlow02.ets`:79:19
- patch: shape=`modify`, changed_lines=15

### Before (local context)
```
    73        Stack({ alignContent: Alignment.TopStart }) {
    74          Column({ space: 2 }) {
    75            WaterFlow({scroller: this.scrollerForWaterFlow}) {
    76              LazyForEach(this.datasource, (item: number) => {
    77                FlowItem() {
    78                  if(item == 66){
    79                    WaterFlow({
    80                      scroller: this.itemScrollerForWaterFlow
    81                    }) {
    82                      LazyForEach(this.datasource, (item: number) => {
    83                        FlowItem() {
    84                          Column() {
    85                            Text('NN' + item).textAlign(TextAlign.End)
```

### After (local context)
```
    73          .id('isHaveSmooth_WaterFlow02').position({x:10, y: 10})
    74        Stack({ alignContent: Alignment.TopStart }) {
    75          Column({ space: 2 }) {
    76            WaterFlow({scroller: this.scrollerForWaterFlow}) {
    77              LazyForEach(this.datasource, (item: number) => {
    78                FlowItem() {
    79                  if(item == 66){
    80                    WaterFlow({
    81                      scroller: this.itemScrollerForWaterFlow
    82                    }) {
    83                      LazyForEach(this.datasource, (item: number) => {
    84                        FlowItem() {
    85                          Column() {
```

### Local diff
```diff
--- before/entry/src/main/ets/MainAbility/pages/ScrollToIndex/ScrollToIndexForWaterFlow02.ets
+++ after/entry/src/main/ets/MainAbility/pages/ScrollToIndex/ScrollToIndexForWaterFlow02.ets
@@ -1,3 +1,4 @@
+    .height('5%')
   }
   build() {
     Column(){
@@ -22,4 +23,3 @@
                           Image('res/waterFlowTest(' + item % 5 + ').jpg')
                             .objectFit(ImageFit.Fill)
                             .width('100%')
-                            .layoutWeight(1)
```

## ab54f4cb1458
- project: `ace_ets_module_nowear_waterflow`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-redundant-state-var` (suggestion)
- file: `entry/src/main/ets/MainAbility/pages/WaterFlowSectionDemo/WaterFlowSectionDemo05.ets`:38:3
- patch: shape=`modify`, changed_lines=9

### Before (local context)
```
    32  }
    33  
    34  @Entry
    35  @Component
    36  struct WaterFlowSectionDemo05 {
    37    @State minSize: number = 80
    38    @State maxSize: number = 180
    39    @State fontSize: number = 24
    40    @State colors: number[] = [0xFFC0CB, 0xDA70D6, 0x6B8E23, 0x6A5ACD, 0x00FFFF, 0x00FF7F]
    41    @State columns: string = '1fr'
    42    scroller: Scroller = new Scroller()
    43    dataCount: number = 7
    44    dataSource: WaterFlowDataSource = new WaterFlowDataSource(this.dataCount)
```

### After (local context)
```
    32      }
    33    }
    34  }
    35  
    36  @Entry
    37  @Component
    38  struct WaterFlowSectionDemo05 {
    39    minSize: number = 80
    40    maxSize: number = 180
    41    fontSize: number = 24
    42    @State colors: number[] = [0xFFC0CB, 0xDA70D6, 0x6B8E23, 0x6A5ACD, 0x00FFFF, 0x00FF7F]
    43    columns: string = '1fr'
    44    scroller: Scroller = new Scroller()
```

### Local diff
```diff
--- before/entry/src/main/ets/MainAbility/pages/WaterFlowSectionDemo/WaterFlowSectionDemo05.ets
+++ after/entry/src/main/ets/MainAbility/pages/WaterFlowSectionDemo/WaterFlowSectionDemo05.ets
@@ -1,3 +1,5 @@
+    Column() {
+      Text('N' + this.item).fontSize(12).height('16')
       Image('res/waterFlowTest (' + this.item % 5 + ').jpg')
         .objectFit(ImageFit.Fill)
         .width('100%')
@@ -9,11 +11,11 @@
 @Entry
 @Component
 struct WaterFlowSectionDemo05 {
-  @State minSize: number = 80
-  @State maxSize: number = 180
-  @State fontSize: number = 24
+  minSize: number = 80
+  maxSize: number = 180
+  fontSize: number = 24
   @State colors: number[] = [0xFFC0CB, 0xDA70D6, 0x6B8E23, 0x6A5ACD, 0x00FFFF, 0x00FF7F]
-  @State columns: string = '1fr'
+  columns: string = '1fr'
   scroller: Scroller = new Scroller()
   dataCount: number = 7
   dataSource: WaterFlowDataSource = new WaterFlowDataSource(this.dataCount)
@@ -21,5 +23,3 @@
 
   twoColumnSection: SectionOptions = {
     itemsCount: 5,
-    crossCount: 1,
-    columnsGap: 0,
```

## d74600e3fac0
- project: `ace_ets_module_nowear_waterflow`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-unchanged-state-var` (suggestion)
- file: `entry/src/main/ets/MainAbility/pages/WaterFlowSectionDemo/WaterFlowSectionDemo12.ets`:38:3
- patch: shape=`modify`, changed_lines=7

### Before (local context)
```
    32  
    33  @Entry
    34  @Component
    35  struct WaterFlowSectionDemo12 {
    36    @State minSize: number = 80
    37    @State maxSize: number = 180
    38    @State fontSize: number = 24
    39    @State colors: number[] = [0xFFC0CB, 0xDA70D6, 0x6B8E23, 0x6A5ACD, 0x00FFFF, 0x00FF7F]
    40    @State columns: string = '1fr'
    41    scroller: Scroller = new Scroller()
    42    dataCount: number = 7
    43    dataSource: WaterFlowDataSource = new WaterFlowDataSource(this.dataCount)
    44    @State sections: WaterFlowSections = new WaterFlowSections()
```

### After (local context)
```
    32  }
    33  
    34  @Entry
    35  @Component
    36  struct WaterFlowSectionDemo12 {
    37    minSize: number = 80
    38    maxSize: number = 180
    39    fontSize: number = 24
    40    @State colors: number[] = [0xFFC0CB, 0xDA70D6, 0x6B8E23, 0x6A5ACD, 0x00FFFF, 0x00FF7F]
    41    columns: string = '1fr'
    42    scroller: Scroller = new Scroller()
    43    dataCount: number = 7
    44    dataSource: WaterFlowDataSource = new WaterFlowDataSource(this.dataCount)
```

### Local diff
```diff
--- before/entry/src/main/ets/MainAbility/pages/WaterFlowSectionDemo/WaterFlowSectionDemo12.ets
+++ after/entry/src/main/ets/MainAbility/pages/WaterFlowSectionDemo/WaterFlowSectionDemo12.ets
@@ -1,3 +1,4 @@
+      Image('res/waterFlowTest (' + this.item % 5 + ').jpg')
         .objectFit(ImageFit.Fill)
         .width('100%')
         .layoutWeight(1)
@@ -8,11 +9,11 @@
 @Entry
 @Component
 struct WaterFlowSectionDemo12 {
-  @State minSize: number = 80
-  @State maxSize: number = 180
-  @State fontSize: number = 24
+  minSize: number = 80
+  maxSize: number = 180
+  fontSize: number = 24
   @State colors: number[] = [0xFFC0CB, 0xDA70D6, 0x6B8E23, 0x6A5ACD, 0x00FFFF, 0x00FF7F]
-  @State columns: string = '1fr'
+  columns: string = '1fr'
   scroller: Scroller = new Scroller()
   dataCount: number = 7
   dataSource: WaterFlowDataSource = new WaterFlowDataSource(this.dataCount)
@@ -22,4 +23,3 @@
     itemsCount: 5,
     crossCount: 3,
     columnsGap: 0,
-    rowsGap: 0,
```

## 30d051a02dbe
- project: `ace_ets_module_nowear_waterflow`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-unchanged-state-var` (suggestion)
- file: `entry/src/main/ets/MainAbility/pages/WaterFlowSectionDemo/WaterFlowSectionDemo14.ets`:36:3
- patch: shape=`modify`, changed_lines=8

### Before (local context)
```
    30    }
    31  }
    32  
    33  @Entry
    34  @Component
    35  struct WaterFlowSectionDemo14 {
    36    @State minSize: number = 80
    37    @State maxSize: number = 180
    38    @State fontSize: number = 24
    39    @State colors: number[] = [0xFFC0CB, 0xDA70D6, 0x6B8E23, 0x6A5ACD, 0x00FFFF, 0x00FF7F]
    40    @State columns: string = '1fr'
    41    scroller: Scroller = new Scroller()
    42    dataCount: number = 7
```

### After (local context)
```
    30          .layoutWeight(1)
    31      }
    32    }
    33  }
    34  
    35  @Entry
    36  @Component
    37  struct WaterFlowSectionDemo14 {
    38    minSize: number = 80
    39    maxSize: number = 180
    40    fontSize: number = 24
    41    @State colors: number[] = [0xFFC0CB, 0xDA70D6, 0x6B8E23, 0x6A5ACD, 0x00FFFF, 0x00FF7F]
    42    columns: string = '1fr'
```

### Local diff
```diff
--- before/entry/src/main/ets/MainAbility/pages/WaterFlowSectionDemo/WaterFlowSectionDemo14.ets
+++ after/entry/src/main/ets/MainAbility/pages/WaterFlowSectionDemo/WaterFlowSectionDemo14.ets
@@ -1,3 +1,5 @@
+  build() {
+    Column() {
       Text('N' + this.item).fontSize(12).height('16')
       Image('res/waterFlowTest (' + this.item % 5 + ').jpg')
         .objectFit(ImageFit.Fill)
@@ -10,16 +12,14 @@
 @Entry
 @Component
 struct WaterFlowSectionDemo14 {
-  @State minSize: number = 80
-  @State maxSize: number = 180
-  @State fontSize: number = 24
+  minSize: number = 80
+  maxSize: number = 180
+  fontSize: number = 24
   @State colors: number[] = [0xFFC0CB, 0xDA70D6, 0x6B8E23, 0x6A5ACD, 0x00FFFF, 0x00FF7F]
-  @State columns: string = '1fr'
+  columns: string = '1fr'
   scroller: Scroller = new Scroller()
   dataCount: number = 7
   dataSource: WaterFlowDataSource = new WaterFlowDataSource(this.dataCount)
   @State sections: WaterFlowSections = new WaterFlowSections()
 
   twoColumnSection: SectionOptions = {
-    itemsCount: 5,
-    crossCount: 3,
```

## c7a7f0024448
- project: `ace_ets_module_nowear_waterflow`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-redundant-state-var` (suggestion)
- file: `entry/src/main/ets/MainAbility/pages/WaterFlowSectionDemo/WaterFlowSectionDemo20.ets`:37:3
- patch: shape=`modify`, changed_lines=9

### Before (local context)
```
    31  }
    32  
    33  @Entry
    34  @Component
    35  struct WaterFlowSectionDemo20 {
    36    @State minSize: number = 80
    37    @State maxSize: number = 180
    38    @State fontSize: number = 24
    39    @State colors: number[] = [0xFFC0CB, 0xDA70D6, 0x6B8E23, 0x6A5ACD, 0x00FFFF, 0x00FF7F]
    40    @State columns: string = '1fr'
    41    scroller: Scroller = new Scroller()
    42    dataCount: number = 7
    43    dataSource: WaterFlowDataSource = new WaterFlowDataSource(this.dataCount)
```

### After (local context)
```
    31      }
    32    }
    33  }
    34  
    35  @Entry
    36  @Component
    37  struct WaterFlowSectionDemo20 {
    38    minSize: number = 80
    39    maxSize: number = 180
    40    fontSize: number = 24
    41    @State colors: number[] = [0xFFC0CB, 0xDA70D6, 0x6B8E23, 0x6A5ACD, 0x00FFFF, 0x00FF7F]
    42    columns: string = '1fr'
    43    scroller: Scroller = new Scroller()
```

### Local diff
```diff
--- before/entry/src/main/ets/MainAbility/pages/WaterFlowSectionDemo/WaterFlowSectionDemo20.ets
+++ after/entry/src/main/ets/MainAbility/pages/WaterFlowSectionDemo/WaterFlowSectionDemo20.ets
@@ -1,3 +1,5 @@
+    Column() {
+      Text('N' + this.item).fontSize(12).height('16')
       Image('res/waterFlowTest (' + this.item % 5 + ').jpg')
         .objectFit(ImageFit.Fill)
         .width('100%')
@@ -9,11 +11,11 @@
 @Entry
 @Component
 struct WaterFlowSectionDemo20 {
-  @State minSize: number = 80
-  @State maxSize: number = 180
-  @State fontSize: number = 24
+  minSize: number = 80
+  maxSize: number = 180
+  fontSize: number = 24
   @State colors: number[] = [0xFFC0CB, 0xDA70D6, 0x6B8E23, 0x6A5ACD, 0x00FFFF, 0x00FF7F]
-  @State columns: string = '1fr'
+  columns: string = '1fr'
   scroller: Scroller = new Scroller()
   dataCount: number = 7
   dataSource: WaterFlowDataSource = new WaterFlowDataSource(this.dataCount)
@@ -21,5 +23,3 @@
 
   twoColumnSection: SectionOptions = {
     itemsCount: 1,
-    crossCount: 1,
-    columnsGap: 0,
```

## 2e4602a69c18
- project: `ace_ets_module_nowear_waterflow`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-use-reusable-component` (warn)
- file: `entry/src/main/ets/MainAbility/pages/WaterFlowSectionDemo/WaterFlowSectionDemo27.ets`:81:13
- patch: shape=`modify`, changed_lines=7

### Before (local context)
```
    75  
    76    build() {
    77      Column({ space: 2 }) {
    78        WaterFlow({ scroller: this.scroller, sections: this.sections }) {
    79          LazyForEach(this.dataSource, (item: number) => {
    80            FlowItem() {
    81              ReusableFlowItem27({ item: item })
    82            }
    83            .key(`WaterFlowSectionDemo27_${item}`)
    84            .width('100%')
    85            .height(100)
    86            .backgroundColor(this.colors[item % 5])
    87          }, (item: string) => item)
```

### After (local context)
```
    75    }
    76  
    77    build() {
    78      Column({ space: 2 }) {
    79        WaterFlow({ scroller: this.scroller, sections: this.sections }) {
    80          LazyForEach(this.dataSource, (item: number) => {
    81            FlowItem() {
    82              ReusableFlowItem27({ item: item })
    83            }
    84            .key(`WaterFlowSectionDemo27_${item}`)
    85            .width('100%')
    86            .height(100)
    87            .backgroundColor(this.colors[item % 5])
```

### Local diff
```diff
--- before/entry/src/main/ets/MainAbility/pages/WaterFlowSectionDemo/WaterFlowSectionDemo27.ets
+++ after/entry/src/main/ets/MainAbility/pages/WaterFlowSectionDemo/WaterFlowSectionDemo27.ets
@@ -1,3 +1,4 @@
+      sectionOptions.push(this.twoColumnSection)
       this.lastSection.itemsCount = this.dataCount - this.twoColumnSection.itemsCount
       sectionOptions.push(this.lastSection)
       break;
@@ -16,10 +17,9 @@
           .width('100%')
           .height(100)
           .backgroundColor(this.colors[item % 5])
-        }, (item: string) => item)
+        }, (item: string) => item).cachedCount(4)
       }
       .columnsTemplate('1fr')
       .columnsGap(10)
       .rowsGap(10)
       .backgroundColor(0xFAEEE0)
-      .width(300)
```

## 28a026450169
- project: `ace_ets_module_nowear_waterflow`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-use-reusable-component` (warn)
- file: `entry/src/main/ets/MainAbility/pages/WaterFlowSectionDemo/WaterFlowSectionDemo31.ets`:81:13
- patch: shape=`modify`, changed_lines=7

### Before (local context)
```
    75  
    76    build() {
    77      Column({ space: 2 }) {
    78        WaterFlow({ scroller: this.scroller, sections: this.sections }) {
    79          LazyForEach(this.dataSource, (item: number) => {
    80            FlowItem() {
    81              ReusableFlowItem31({ item: item })
    82            }
    83            .key(`WaterFlowSectionDemo31_${item}`)
    84            .width('100%')
    85            .height(100)
    86            .backgroundColor(this.colors[item % 5])
    87          }, (item: string) => item)
```

### After (local context)
```
    75    }
    76  
    77    build() {
    78      Column({ space: 2 }) {
    79        WaterFlow({ scroller: this.scroller, sections: this.sections }) {
    80          LazyForEach(this.dataSource, (item: number) => {
    81            FlowItem() {
    82              ReusableFlowItem31({ item: item })
    83            }
    84            .key(`WaterFlowSectionDemo31_${item}`)
    85            .width('100%')
    86            .height(100)
    87            .backgroundColor(this.colors[item % 5])
```

### Local diff
```diff
--- before/entry/src/main/ets/MainAbility/pages/WaterFlowSectionDemo/WaterFlowSectionDemo31.ets
+++ after/entry/src/main/ets/MainAbility/pages/WaterFlowSectionDemo/WaterFlowSectionDemo31.ets
@@ -1,3 +1,4 @@
+      sectionOptions.push(this.twoColumnSection)
       this.lastSection.itemsCount = this.dataCount - this.twoColumnSection.itemsCount
       sectionOptions.push(this.lastSection)
       break;
@@ -16,10 +17,9 @@
           .width('100%')
           .height(100)
           .backgroundColor(this.colors[item % 5])
-        }, (item: string) => item)
+        }, (item: string) => item).cachedCount(4)
       }
       .columnsTemplate('1fr')
       .columnsGap(10)
       .rowsGap(10)
       .backgroundColor(0xFAEEE0)
-      .width(300)
```

## 3507e1578df6
- project: `ace_ets_module_nowear_waterflow`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-unchanged-state-var` (suggestion)
- file: `entry/src/main/ets/MainAbility/pages/WaterFlowSectionDemo/WaterFlowSectionDemo33.ets`:36:3
- patch: shape=`modify`, changed_lines=8

### Before (local context)
```
    30    }
    31  }
    32  
    33  @Entry
    34  @Component
    35  struct WaterFlowSectionDemo33 {
    36    @State minSize: number = 80
    37    @State maxSize: number = 180
    38    @State fontSize: number = 24
    39    @State colors: number[] = [0xFFC0CB, 0xDA70D6, 0x6B8E23, 0x6A5ACD, 0x00FFFF, 0x00FF7F]
    40    @State columns: string = '1fr'
    41    scroller: Scroller = new Scroller()
    42    dataCount: number = 40
```

### After (local context)
```
    30      }
    31    }
    32  }
    33  
    34  @Entry
    35  @Component
    36  struct WaterFlowSectionDemo33 {
    37    minSize: number = 80
    38    maxSize: number = 180
    39    fontSize: number = 24
    40    @State colors: number[] = [0xFFC0CB, 0xDA70D6, 0x6B8E23, 0x6A5ACD, 0x00FFFF, 0x00FF7F]
    41    columns: string = '1fr'
    42    scroller: Scroller = new Scroller()
```

### Local diff
```diff
--- before/entry/src/main/ets/MainAbility/pages/WaterFlowSectionDemo/WaterFlowSectionDemo33.ets
+++ after/entry/src/main/ets/MainAbility/pages/WaterFlowSectionDemo/WaterFlowSectionDemo33.ets
@@ -1,3 +1,4 @@
+    Column() {
       Text('N' + this.item).fontSize(12).height('16')
       Image('res/waterFlowTest (' + this.item % 5 + ').jpg')
         .objectFit(ImageFit.Fill)
@@ -10,11 +11,11 @@
 @Entry
 @Component
 struct WaterFlowSectionDemo33 {
-  @State minSize: number = 80
-  @State maxSize: number = 180
-  @State fontSize: number = 24
+  minSize: number = 80
+  maxSize: number = 180
+  fontSize: number = 24
   @State colors: number[] = [0xFFC0CB, 0xDA70D6, 0x6B8E23, 0x6A5ACD, 0x00FFFF, 0x00FF7F]
-  @State columns: string = '1fr'
+  columns: string = '1fr'
   scroller: Scroller = new Scroller()
   dataCount: number = 40
   dataSource: WaterFlowDataSource = new WaterFlowDataSource(this.dataCount)
@@ -22,4 +23,3 @@
   oneColumnSection: SectionOptions = {
     itemsCount: 8,
     crossCount: 1,
-    columnsGap: 0,
```

## af7c2084a25a
- project: `ace_ets_module_nowear_waterflow`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-use-reusable-component` (warn)
- file: `entry/src/main/ets/MainAbility/pages/sectionOptions/SectionsFalse.ets`:80:15
- patch: shape=`modify`, changed_lines=21

### Before (local context)
```
    74          })
    75  
    76        WaterFlow({ scroller: this.scroller, sections: this.sections }) {
    77          LazyForEach(this.dataSource, (item: number) => {
    78            FlowItem() {
    79              Column() {
    80                Text('N' + item).fontSize(12).height('16')
    81              }
    82            }
    83            .width('100%')
    84            .height(this.itemHeightArray[item % 100])
    85            .backgroundColor(this.colors[item % this.colors.length])
    86          }, (item: string) => item)
```

### After (local context)
```
    74  
    75            let spliceFalse: boolean = this.sections.splice(0, oldLength, [newSection]);
    76            AppStorage.SetOrCreate('spliceFalse', spliceFalse);
    77  
    78            let pushFalse: boolean = this.sections.push(newSection);
    79            AppStorage.SetOrCreate('pushFalse', pushFalse);
    80  
    81            let updateFalse:boolean = this.sections.update(0, newSection);
    82            AppStorage.SetOrCreate('updateFalse', updateFalse);
    83  
    84            const valuesNull: SectionOptions[] = this.sections.values();
    85            AppStorage.SetOrCreate('valuesNull', valuesNull);
    86            let valuesLength:number = this.sections.length();
```

### Local diff
```diff
--- before/entry/src/main/ets/MainAbility/pages/sectionOptions/SectionsFalse.ets
+++ after/entry/src/main/ets/MainAbility/pages/sectionOptions/SectionsFalse.ets
@@ -1,3 +1,17 @@
+        .onClick(() => {
+          let totalCount: number = this.dataSource.totalCount();
+          let newSection: SectionOptions = {
+            itemsCount: -1,
+          };
+          let oldLength: number = this.sections.length();
+
+          let spliceFalse: boolean = this.sections.splice(0, oldLength, [newSection]);
+          AppStorage.SetOrCreate('spliceFalse', spliceFalse);
+
+          let pushFalse: boolean = this.sections.push(newSection);
+          AppStorage.SetOrCreate('pushFalse', pushFalse);
+
+          let updateFalse:boolean = this.sections.update(0, newSection);
           AppStorage.SetOrCreate('updateFalse', updateFalse);
 
           const valuesNull: SectionOptions[] = this.sections.values();
@@ -7,19 +21,5 @@
         })
 
       WaterFlow({ scroller: this.scroller, sections: this.sections }) {
-        LazyForEach(this.dataSource, (item: number) => {
+        LazyForEach(this.dataSource, (item: number, index) => {
           FlowItem() {
-            Column() {
-              Text('N' + item).fontSize(12).height('16')
-            }
-          }
-          .width('100%')
-          .height(this.itemHeightArray[item % 100])
-          .backgroundColor(this.colors[item % this.colors.length])
-        }, (item: string) => item)
-      }
-      .width('90%')
-      .height('50%')
-      .columnsTemplate('1fr 1fr')
-      .columnsGap(10)
-      .rowsGap(5)
```

## 804147e5de8c
- project: `ace_ets_module_nowear_waterflow`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-container-without-property` (suggestion)
- file: `entry/src/main/ets/MainAbility/pages/waterFlowCallback/WaterFlowFlingCallback.ets`:80:15
- patch: shape=`modify`, changed_lines=21

### Before (local context)
```
    74            })
    75            .width('100%')
    76  		  
    77          WaterFlow({ scroller: this.scroller }) {
    78            LazyForEach(this.dataSource, (item: number) => {
    79              FlowItem() {
    80                Column() {
    81                  Text('N' + item).fontSize(12).height('16')
    82                }
    83              }
    84              .width('100%')
    85              .height(this.itemHeightArray[item % 100])
    86              .backgroundColor(this.colors[item % 5])
```

### After (local context)
```
    74            .width('100%')
    75          Text('' + this.scroller_fling_1).fontColor(0x000000)
    76            .fontSize(16).width('90%').key('key_waterflow_fling_text1')
    77          Text('' + this.scroller_fling_2).fontColor(0x000000)
    78            .fontSize(16).width('90%').key('key_waterflow_fling_text2')
    79          Button('更换速度 -1000')
    80            .id('btn_waterflow_fling_02')
    81            .onClick(() => { 
    82              this.scroller.fling(-1000)
    83              console.info('WaterFlowFlingCallbackTest_0200 start to emit action state')
    84              this.scroller_fling_2 = '-1000'
    85            })
    86            .width('100%')
```

### Local diff
```diff
--- before/entry/src/main/ets/MainAbility/pages/waterFlowCallback/WaterFlowFlingCallback.ets
+++ after/entry/src/main/ets/MainAbility/pages/waterFlowCallback/WaterFlowFlingCallback.ets
@@ -1,3 +1,14 @@
+          .id('btn_waterflow_fling_01')
+          .onClick(() => { 
+            this.scroller.fling(1000)
+            console.info('WaterFlowFlingCallbackTest_0100 start to emit action state')
+            this.scroller_fling_1 = '1000'
+          })
+          .width('100%')
+        Text('' + this.scroller_fling_1).fontColor(0x000000)
+          .fontSize(16).width('90%').key('key_waterflow_fling_text1')
+        Text('' + this.scroller_fling_2).fontColor(0x000000)
+          .fontSize(16).width('90%').key('key_waterflow_fling_text2')
         Button('更换速度 -1000')
           .id('btn_waterflow_fling_02')
           .onClick(() => { 
@@ -10,16 +21,5 @@
         WaterFlow({ scroller: this.scroller }) {
           LazyForEach(this.dataSource, (item: number) => {
             FlowItem() {
-              Column() {
-                Text('N' + item).fontSize(12).height('16')
-              }
-            }
-            .width('100%')
-            .height(this.itemHeightArray[item % 100])
-            .backgroundColor(this.colors[item % 5])
-          }, (item: string) => item)
-        }
-        .columnsTemplate('1fr')
-        .columnsGap(10)
-        .rowsGap(5)
-        .padding({ left: 5 })
+
+              WaterFlowItemComponent({
```

## 51724ffb4167
- project: `ace_ets_module_nowear_waterflow`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-unchanged-state-var` (suggestion)
- file: `entry/src/main/ets/MainAbility/pages/waterFlowCallback/WaterFlowScrollEdgeCallback.ets`:22:3
- patch: shape=`modify`, changed_lines=22

### Before (local context)
```
    16  import { WaterFlowDataSource } from '../WaterFlowDataSource'
    17  
    18  @Entry
    19  @Component
    20  struct WaterFlowScrollEdgeCallback {
    21    @State minSize: number = 80
    22    @State maxSize: number = 180
    23    @State clickFlag: number = 0
    24    @State colors: number[] = [0xFFC0CB, 0xDA70D6, 0x6B8E23, 0x6A5ACD, 0x00FFFF, 0x00FF7F]
    25    dataSource: WaterFlowDataSource = new WaterFlowDataSource()
    26    private itemWidthArray: number[] = []
    27    private itemHeightArray: number[] = []
    28    @State numbers: String[] = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
```

### After (local context)
```
    16  import { WaterFlowDataSource } from '../WaterFlowDataSource'
    17  
    18  @Reusable
    19  @Component
    20  struct WaterFlowItemComponent {
    21    item: number = 0
    22  
    23    build() {
    24      Text('N' + this.item)
    25        .fontSize(12)
    26        .height('16')
    27    }
    28  }
```

### Local diff
```diff
--- before/entry/src/main/ets/MainAbility/pages/waterFlowCallback/WaterFlowScrollEdgeCallback.ets
+++ after/entry/src/main/ets/MainAbility/pages/waterFlowCallback/WaterFlowScrollEdgeCallback.ets
@@ -6,20 +6,20 @@
 import events_emitter from '@ohos.events.emitter'
 import { WaterFlowDataSource } from '../WaterFlowDataSource'
 
+@Reusable
+@Component
+struct WaterFlowItemComponent {
+  item: number = 0
+
+  build() {
+    Text('N' + this.item)
+      .fontSize(12)
+      .height('16')
+  }
+}
+
 @Entry
 @Component
 struct WaterFlowScrollEdgeCallback {
-  @State minSize: number = 80
-  @State maxSize: number = 180
-  @State clickFlag: number = 0
-  @State colors: number[] = [0xFFC0CB, 0xDA70D6, 0x6B8E23, 0x6A5ACD, 0x00FFFF, 0x00FF7F]
-  dataSource: WaterFlowDataSource = new WaterFlowDataSource()
-  private itemWidthArray: number[] = []
-  private itemHeightArray: number[] = []
-  @State numbers: String[] = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
-  scroller: Scroller = new Scroller()
-  @State text1:string = ''
-
-  // 计算FlowItem宽/高
-  getSize(){
-    let ret = Math.floor(Math.random() * this.maxSize)
+  minSize: number = 80
+  maxSize: number = 180
```

## ca59dcf1a748
- project: `ace_ets_module_nowear_waterflow`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-set-cache-count-for-lazyforeach-grid` (suggestion)
- file: `entry/src/main/ets/MainAbility/pages/waterFlowCallback/WaterFlowScrollEdgeCallback.ets`:63:9
- patch: shape=`modify`, changed_lines=22

### Before (local context)
```
    57            .id('WaterFlowScrollEdgeCallback01')
    58            .onClick(() => { // 这里
    59              this.scroller.scrollEdge(Edge.Bottom, { velocity: 0 })
    60              this.text1 = 'ScrollEdgeSuccess'
    61            })
    62            .width('100%')
    63          WaterFlow({ scroller: this.scroller }) {
    64            LazyForEach(this.dataSource, (item: number) => {
    65              FlowItem() {
    66                Column() {
    67                  Text('N' + item).fontSize(12).height('16')
    68                }
    69              }
```

### After (local context)
```
    57  
    58    aboutToAppear(){
    59      this.setItemSizeArray()
    60    }
    61  
    62    build(){
    63      Scroll() {
    64        Column({ space: 2 }) {
    65          Text('Scroller组件绑定至平铺的WaterFlow容器组件').fontColor(0x000000)
    66            .fontSize(16).width('90%')
    67          Text(this.text1).id('WaterFlowTextByScrollEdge01')
    68          Button('点我滚动')
    69            .id('WaterFlowScrollEdgeCallback01')
```

### Local diff
```diff
--- before/entry/src/main/ets/MainAbility/pages/waterFlowCallback/WaterFlowScrollEdgeCallback.ets
+++ after/entry/src/main/ets/MainAbility/pages/waterFlowCallback/WaterFlowScrollEdgeCallback.ets
@@ -1,3 +1,15 @@
+  setItemSizeArray(){
+    for (let i = 0; i < 100; i++) {
+      this.itemWidthArray.push(this.getSize())
+      this.itemHeightArray.push(this.getSize())
+    }
+  }
+
+  aboutToAppear(){
+    this.setItemSizeArray()
+  }
+
+  build(){
     Scroll() {
       Column({ space: 2 }) {
         Text('Scroller组件绑定至平铺的WaterFlow容器组件').fontColor(0x000000)
@@ -11,15 +23,3 @@
           })
           .width('100%')
         WaterFlow({ scroller: this.scroller }) {
-          LazyForEach(this.dataSource, (item: number) => {
-            FlowItem() {
-              Column() {
-                Text('N' + item).fontSize(12).height('16')
-              }
-            }
-            .width('100%')
-            .height(this.itemHeightArray[item % 100])
-            .backgroundColor(this.colors[item % 5])
-          }, (item: string) => item)
-        }
-        .columnsTemplate('1fr')
```

## f2480231b8fa
- project: `ace_ets_module_nowear_waterflow`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-unchanged-state-var` (suggestion)
- file: `entry/src/main/ets/MainAbility/pages/waterFlowCallback/waterFlowAniTrue.ets`:19:3
- patch: shape=`modify`, changed_lines=18

### Before (local context)
```
    13   * limitations under the License.
    14   */
    15  import { WaterFlowDataSource } from '../WaterFlowDataSource'
    16  @Entry
    17  @Component
    18  struct waterFlowNextAnimationExample {
    19    @State minSize: number = 50
    20    @State maxSize: number = 100
    21    @State padding1: number = 0
    22    scroller: Scroller = new Scroller()
    23    @State colors: number[] = [0xFFC0CB, 0xDA70D6, 0x6B8E23, 0x6A5ACD, 0x00FFFF, 0x00FF7F]
    24    dataSource: WaterFlowDataSource = new WaterFlowDataSource()
    25    @State onText:string = 'WaterFlow: '
```

### After (local context)
```
    13   * limitations under the License.
    14   */
    15  import { WaterFlowDataSource } from '../WaterFlowDataSource'
    16  @Reusable
    17  @Component
    18  struct WaterFlowItemComponent {
    19    item: number = 0
    20  
    21    build() {
    22      Text('N' + this.item)
    23        .fontSize(12)
    24        .height('16')
    25    }
```

### Local diff
```diff
--- before/entry/src/main/ets/MainAbility/pages/waterFlowCallback/waterFlowAniTrue.ets
+++ after/entry/src/main/ets/MainAbility/pages/waterFlowCallback/waterFlowAniTrue.ets
@@ -7,19 +7,19 @@
  * limitations under the License.
  */
 import { WaterFlowDataSource } from '../WaterFlowDataSource'
+@Reusable
+@Component
+struct WaterFlowItemComponent {
+  item: number = 0
+
+  build() {
+    Text('N' + this.item)
+      .fontSize(12)
+      .height('16')
+  }
+}
 @Entry
 @Component
 struct waterFlowNextAnimationExample {
-  @State minSize: number = 50
-  @State maxSize: number = 100
-  @State padding1: number = 0
-  scroller: Scroller = new Scroller()
-  @State colors: number[] = [0xFFC0CB, 0xDA70D6, 0x6B8E23, 0x6A5ACD, 0x00FFFF, 0x00FF7F]
-  dataSource: WaterFlowDataSource = new WaterFlowDataSource()
-  @State onText:string = 'WaterFlow: '
-  @State nextBln:boolean = false
-  private itemWidthArray: number[] = []
-  private itemHeightArray: number[] = []
-
-  // 计算FlowItem宽/高
-  getSize() {
+  minSize: number = 50
+  maxSize: number = 100
```

## 06b0bf3d8060
- project: `ace_ets_module_scroll_nowear_api12`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-set-cache-count-for-lazyforeach-grid` (suggestion)
- file: `entry/src/main/ets/MainAbility/pages/WaterFlow/SectionsFalse.ets`:76:7
- patch: shape=`modify`, changed_lines=18

### Before (local context)
```
    70            const valuesNull: SectionOptions[] = this.sections.values();
    71            AppStorage.SetOrCreate('valuesNull', valuesNull);
    72            let valuesLength:number = this.sections.length();
    73            console.log('values length: ' + valuesLength);
    74          })
    75  
    76        WaterFlow({ scroller: this.scroller, sections: this.sections }) {
    77          LazyForEach(this.dataSource, (item: number) => {
    78            FlowItem() {
    79              Column() {
    80                Text('N' + item).fontSize(12).height('16')
    81              }
    82            }
```

### After (local context)
```
    70            };
    71            let oldLength: number = this.sections.length();
    72  
    73            let spliceFalse: boolean = this.sections.splice(0, oldLength, [newSection]);
    74            AppStorage.SetOrCreate('spliceFalse', spliceFalse);
    75  
    76            let pushFalse: boolean = this.sections.push(newSection);
    77            AppStorage.SetOrCreate('pushFalse', pushFalse);
    78  
    79            let updateFalse:boolean = this.sections.update(0, newSection);
    80            AppStorage.SetOrCreate('updateFalse', updateFalse);
    81  
    82            const valuesNull: SectionOptions[] = this.sections.values();
```

### Local diff
```diff
--- before/entry/src/main/ets/MainAbility/pages/WaterFlow/SectionsFalse.ets
+++ after/entry/src/main/ets/MainAbility/pages/WaterFlow/SectionsFalse.ets
@@ -1,3 +1,15 @@
+        .id('sections_false')
+        .margin(10)
+        .onClick(() => {
+          let totalCount: number = this.dataSource.totalCount();
+          let newSection: SectionOptions = {
+            itemsCount: -1,
+          };
+          let oldLength: number = this.sections.length();
+
+          let spliceFalse: boolean = this.sections.splice(0, oldLength, [newSection]);
+          AppStorage.SetOrCreate('spliceFalse', spliceFalse);
+
           let pushFalse: boolean = this.sections.push(newSection);
           AppStorage.SetOrCreate('pushFalse', pushFalse);
 
@@ -11,15 +23,3 @@
         })
 
       WaterFlow({ scroller: this.scroller, sections: this.sections }) {
-        LazyForEach(this.dataSource, (item: number) => {
-          FlowItem() {
-            Column() {
-              Text('N' + item).fontSize(12).height('16')
-            }
-          }
-          .width('100%')
-          .height(this.itemHeightArray[item % 100])
-          .backgroundColor(this.colors[item % this.colors.length])
-        }, (item: string) => item)
-      }
-      .width('90%')
```

## bf7be09a28f5
- project: `ace_ets_module_scroll_nowear_api12`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-redundant-state-var` (suggestion)
- file: `entry/src/main/ets/MainAbility/pages/WaterFlow/WaterFlowSectionDemo/WaterFlowSectionDemo06.ets`:37:3
- patch: shape=`modify`, changed_lines=9

### Before (local context)
```
    31    }
    32  }
    33  
    34  @Entry
    35  @Component
    36  struct WaterFlowSectionDemo06 {
    37    @State minSize: number = 80
    38    @State maxSize: number = 180
    39    @State fontSize: number = 24
    40    @State colors: number[] = [0xFFC0CB, 0xDA70D6, 0x6B8E23, 0x6A5ACD, 0x00FFFF, 0x00FF7F]
    41    @State columns: string = '1fr'
    42    scroller: Scroller = new Scroller()
    43    dataCount: number = 7
```

### After (local context)
```
    31          .layoutWeight(1)
    32      }
    33    }
    34  }
    35  
    36  @Entry
    37  @Component
    38  struct WaterFlowSectionDemo06 {
    39    minSize: number = 80
    40    maxSize: number = 180
    41    fontSize: number = 24
    42    @State colors: number[] = [0xFFC0CB, 0xDA70D6, 0x6B8E23, 0x6A5ACD, 0x00FFFF, 0x00FF7F]
    43    columns: string = '1fr'
```

### Local diff
```diff
--- before/entry/src/main/ets/MainAbility/pages/WaterFlow/WaterFlowSectionDemo/WaterFlowSectionDemo06.ets
+++ after/entry/src/main/ets/MainAbility/pages/WaterFlow/WaterFlowSectionDemo/WaterFlowSectionDemo06.ets
@@ -1,3 +1,5 @@
+  build() {
+    Column() {
       Text("N" + this.item).fontSize(12).height('16')
       Image('res/waterFlowTest (' + this.item % 5 + ').jpg')
         .objectFit(ImageFit.Fill)
@@ -10,16 +12,14 @@
 @Entry
 @Component
 struct WaterFlowSectionDemo06 {
-  @State minSize: number = 80
-  @State maxSize: number = 180
-  @State fontSize: number = 24
+  minSize: number = 80
+  maxSize: number = 180
+  fontSize: number = 24
   @State colors: number[] = [0xFFC0CB, 0xDA70D6, 0x6B8E23, 0x6A5ACD, 0x00FFFF, 0x00FF7F]
-  @State columns: string = '1fr'
+  columns: string = '1fr'
   scroller: Scroller = new Scroller()
   dataCount: number = 7
   dataSource: WaterFlowDataSource = new WaterFlowDataSource(this.dataCount)
   @State sections: WaterFlowSections = new WaterFlowSections()
 
   twoColumnSection: SectionOptions = {
-    itemsCount: 5,
-    crossCount: 2,
```

## 1ebcbc550c21
- project: `ace_ets_module_scroll_nowear_api12`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-use-reusable-component` (warn)
- file: `entry/src/main/ets/MainAbility/pages/WaterFlow/WaterFlowSectionDemo/WaterFlowSectionDemo11.ets`:80:13
- patch: shape=`modify`, changed_lines=6

### Before (local context)
```
    74  
    75    build() {
    76      Column({ space: 2 }) {
    77        WaterFlow({ scroller: this.scroller, sections: this.sections }) {
    78          LazyForEach(this.dataSource, (item: number) => {
    79            FlowItem() {
    80              ReusableFlowItem11({ item: item })
    81            }
    82            .key(`WaterFlowSectionDemo11_${item}`)
    83            .width('100%')
    84            .height(100)
    85            .backgroundColor(this.colors[item % 5])
    86          }, (item: string) => item)
```

### After (local context)
```
    74  
    75    build() {
    76      Column({ space: 2 }) {
    77        WaterFlow({ scroller: this.scroller, sections: this.sections }) {
    78          LazyForEach(this.dataSource, (item: number) => {
    79            FlowItem() {
    80              ReusableFlowItem11({ item: item })
    81            }
    82            .key(`WaterFlowSectionDemo11_${item}`)
    83            .width('100%')
    84            .height(100)
    85            .backgroundColor(this.colors[item % 5])
    86          }, (item: string) => item).cachedCount(4)
```

### Local diff
```diff
--- before/entry/src/main/ets/MainAbility/pages/WaterFlow/WaterFlowSectionDemo/WaterFlowSectionDemo11.ets
+++ after/entry/src/main/ets/MainAbility/pages/WaterFlow/WaterFlowSectionDemo/WaterFlowSectionDemo11.ets
@@ -16,7 +16,7 @@
           .width('100%')
           .height(100)
           .backgroundColor(this.colors[item % 5])
-        }, (item: string) => item)
+        }, (item: string) => item).cachedCount(4)
       }
       .columnsTemplate('1fr')
       .columnsGap(10)
```

## b3123c72a944
- project: `ace_ets_module_scroll_nowear_api12`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-unchanged-state-var` (suggestion)
- file: `entry/src/main/ets/MainAbility/pages/WaterFlow/WaterFlowSectionDemo/WaterFlowSectionDemo12.ets`:37:3
- patch: shape=`modify`, changed_lines=8

### Before (local context)
```
    31  }
    32  
    33  @Entry
    34  @Component
    35  struct WaterFlowSectionDemo12 {
    36    @State minSize: number = 80
    37    @State maxSize: number = 180
    38    @State fontSize: number = 24
    39    @State colors: number[] = [0xFFC0CB, 0xDA70D6, 0x6B8E23, 0x6A5ACD, 0x00FFFF, 0x00FF7F]
    40    @State columns: string = '1fr'
    41    scroller: Scroller = new Scroller()
    42    dataCount: number = 7
    43    dataSource: WaterFlowDataSource = new WaterFlowDataSource(this.dataCount)
```

### After (local context)
```
    31    }
    32  }
    33  
    34  @Entry
    35  @Component
    36  struct WaterFlowSectionDemo12 {
    37    minSize: number = 80
    38    maxSize: number = 180
    39    fontSize: number = 24
    40    @State colors: number[] = [0xFFC0CB, 0xDA70D6, 0x6B8E23, 0x6A5ACD, 0x00FFFF, 0x00FF7F]
    41    columns: string = '1fr'
    42    scroller: Scroller = new Scroller()
    43    dataCount: number = 7
```

### Local diff
```diff
--- before/entry/src/main/ets/MainAbility/pages/WaterFlow/WaterFlowSectionDemo/WaterFlowSectionDemo12.ets
+++ after/entry/src/main/ets/MainAbility/pages/WaterFlow/WaterFlowSectionDemo/WaterFlowSectionDemo12.ets
@@ -1,3 +1,4 @@
+      Text("N" + this.item).fontSize(12).height('16')
       Image('res/waterFlowTest (' + this.item % 5 + ').jpg')
         .objectFit(ImageFit.Fill)
         .width('100%')
@@ -9,11 +10,11 @@
 @Entry
 @Component
 struct WaterFlowSectionDemo12 {
-  @State minSize: number = 80
-  @State maxSize: number = 180
-  @State fontSize: number = 24
+  minSize: number = 80
+  maxSize: number = 180
+  fontSize: number = 24
   @State colors: number[] = [0xFFC0CB, 0xDA70D6, 0x6B8E23, 0x6A5ACD, 0x00FFFF, 0x00FF7F]
-  @State columns: string = '1fr'
+  columns: string = '1fr'
   scroller: Scroller = new Scroller()
   dataCount: number = 7
   dataSource: WaterFlowDataSource = new WaterFlowDataSource(this.dataCount)
@@ -22,4 +23,3 @@
   twoColumnSection: SectionOptions = {
     itemsCount: 5,
     crossCount: 3,
-    columnsGap: 0,
```

## 9ea3b536343b
- project: `ace_ets_module_scroll_nowear_api12`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-use-reusable-component` (warn)
- file: `entry/src/main/ets/MainAbility/pages/WaterFlow/WaterFlowSectionDemo/WaterFlowSectionDemo25.ets`:81:13
- patch: shape=`modify`, changed_lines=7

### Before (local context)
```
    75  
    76    build() {
    77      Column({ space: 2 }) {
    78        WaterFlow({ scroller: this.scroller, sections: this.sections }) {
    79          LazyForEach(this.dataSource, (item: number) => {
    80            FlowItem() {
    81              ReusableFlowItem25({ item: item })
    82            }
    83            .key(`WaterFlowSectionDemo25_${item}`)
    84            .width('100%')
    85            .height(100)
    86            .backgroundColor(this.colors[item % 5])
    87          }, (item: string) => item)
```

### After (local context)
```
    75    }
    76  
    77    build() {
    78      Column({ space: 2 }) {
    79        WaterFlow({ scroller: this.scroller, sections: this.sections }) {
    80          LazyForEach(this.dataSource, (item: number) => {
    81            FlowItem() {
    82              ReusableFlowItem25({ item: item })
    83            }
    84            .key(`WaterFlowSectionDemo25_${item}`)
    85            .width('100%')
    86            .height(100)
    87            .backgroundColor(this.colors[item % 5])
```

### Local diff
```diff
--- before/entry/src/main/ets/MainAbility/pages/WaterFlow/WaterFlowSectionDemo/WaterFlowSectionDemo25.ets
+++ after/entry/src/main/ets/MainAbility/pages/WaterFlow/WaterFlowSectionDemo/WaterFlowSectionDemo25.ets
@@ -1,3 +1,4 @@
+      sectionOptions.push(this.twoColumnSection)
       this.lastSection.itemsCount = this.dataCount - this.twoColumnSection.itemsCount
       sectionOptions.push(this.lastSection)
       break;
@@ -16,10 +17,9 @@
           .width('100%')
           .height(100)
           .backgroundColor(this.colors[item % 5])
-        }, (item: string) => item)
+        }, (item: string) => item).cachedCount(4)
       }
       .columnsTemplate('1fr')
       .columnsGap(10)
       .rowsGap(10)
       .backgroundColor(0xFAEEE0)
-      .width(300)
```

## 9f49df437e7f
- project: `ace_ets_module_scroll_nowear_api12`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-redundant-state-var` (suggestion)
- file: `entry/src/main/ets/MainAbility/pages/WaterFlow/WaterFlowSectionDemo/WaterFlowSectionDemo42.ets`:37:3
- patch: shape=`modify`, changed_lines=7

### Before (local context)
```
    31  }
    32  
    33  @Entry
    34  @Component
    35  struct WaterFlowSectionDemo42 {
    36    @State minSize: number = 80
    37    @State maxSize: number = 180
    38    @State fontSize: number = 24
    39    @State colors: number[] = [0xFFC0CB, 0xDA70D6, 0x6B8E23, 0x6A5ACD, 0x00FFFF, 0x00FF7F]
    40    @State columns: string = '1fr'
    41    scroller: Scroller = new Scroller()
    42    dataCount: number = 40
    43    dataSource: WaterFlowDataSource = new WaterFlowDataSource(this.dataCount)
```

### After (local context)
```
    31    }
    32  }
    33  
    34  @Entry
    35  @Component
    36  struct WaterFlowSectionDemo42 {
    37    minSize: number = 80
    38    maxSize: number = 180
    39    fontSize: number = 24
    40    @State colors: number[] = [0xFFC0CB, 0xDA70D6, 0x6B8E23, 0x6A5ACD, 0x00FFFF, 0x00FF7F]
    41    columns: string = '1fr'
    42    scroller: Scroller = new Scroller()
    43    dataCount: number = 40
```

### Local diff
```diff
--- before/entry/src/main/ets/MainAbility/pages/WaterFlow/WaterFlowSectionDemo/WaterFlowSectionDemo42.ets
+++ after/entry/src/main/ets/MainAbility/pages/WaterFlow/WaterFlowSectionDemo/WaterFlowSectionDemo42.ets
@@ -1,3 +1,4 @@
+      Text("N" + this.item).fontSize(12).height('16')
       Image('res/waterFlowTest (' + this.item % 5 + ').jpg')
         .objectFit(ImageFit.Fill)
         .width('100%')
@@ -9,11 +10,11 @@
 @Entry
 @Component
 struct WaterFlowSectionDemo42 {
-  @State minSize: number = 80
-  @State maxSize: number = 180
-  @State fontSize: number = 24
+  minSize: number = 80
+  maxSize: number = 180
+  fontSize: number = 24
   @State colors: number[] = [0xFFC0CB, 0xDA70D6, 0x6B8E23, 0x6A5ACD, 0x00FFFF, 0x00FF7F]
-  @State columns: string = '1fr'
+  columns: string = '1fr'
   scroller: Scroller = new Scroller()
   dataCount: number = 40
   dataSource: WaterFlowDataSource = new WaterFlowDataSource(this.dataCount)
@@ -22,4 +23,3 @@
     itemsCount: 8,
     crossCount: 1,
     columnsGap: 0,
-    rowsGap: 0,
```

## b9430c7faa74
- project: `ace_ets_module_scroll_nowear_api12`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-container-without-property` (suggestion)
- file: `entry/src/main/ets/MainAbility/pages/WaterFlow/onScrollIndexFlow1.ets`:58:13
- patch: shape=`modify`, changed_lines=19

### Before (local context)
```
    52            .id('onScrollIndex_flow')
    53        }.height('10%')
    54  
    55        WaterFlow({ scroller: this.scroller }) {
    56          LazyForEach(this.dataSource, (item: number) => {
    57            FlowItem() {
    58              Column() {
    59                Text('N' + item).fontSize(12).height('16')
    60              }
    61            }
    62            .width('100%')
    63            .height(this.itemHeightArray[item % 100])
    64            .backgroundColor(this.colors[item % 5])
```

### After (local context)
```
    52        this.itemHeightArray.push(this.getSize())
    53      }
    54    }
    55  
    56    aboutToAppear() {
    57      this.setItemSizeArray()
    58    }
    59  
    60    build() {
    61      Column({ space: 5 }) {
    62        Column(){
    63          Text(this.onScrollIndex)
    64            .id('onScrollIndex_flow')
```

### Local diff
```diff
--- before/entry/src/main/ets/MainAbility/pages/WaterFlow/onScrollIndexFlow1.ets
+++ after/entry/src/main/ets/MainAbility/pages/WaterFlow/onScrollIndexFlow1.ets
@@ -1,3 +1,15 @@
+  }
+
+  // 设置FlowItem宽/高数组
+  setItemSizeArray() {
+    for (let i = 0; i < 100; i++) {
+      this.itemWidthArray.push(this.getSize())
+      this.itemHeightArray.push(this.getSize())
+    }
+  }
+
+  aboutToAppear() {
+    this.setItemSizeArray()
   }
 
   build() {
@@ -10,16 +22,4 @@
       WaterFlow({ scroller: this.scroller }) {
         LazyForEach(this.dataSource, (item: number) => {
           FlowItem() {
-            Column() {
-              Text('N' + item).fontSize(12).height('16')
-            }
-          }
-          .width('100%')
-          .height(this.itemHeightArray[item % 100])
-          .backgroundColor(this.colors[item % 5])
-        }, (item: string) => item)
-      }
-      .id('onScrollIndexFlow1')
-      .rowsTemplate('1fr 1fr 1fr')
-      .direction(Direction.Rtl)
-      .columnsGap(10)
+
```

## ea1f983a73fa
- project: `ace_ets_module_scroll_nowear_api12`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-unchanged-state-var` (suggestion)
- file: `entry/src/main/ets/MainAbility/pages/WaterFlow/waterFlow3.ets`:19:3
- patch: shape=`modify`, changed_lines=22

### Before (local context)
```
    13   * limitations under the License.
    14   */
    15  import { WaterFlowDataSource } from './WaterFlowDataSource'
    16  @Entry
    17  @Component
    18  struct WaterFlow3 {
    19    @State minSize: number = 80
    20    @State maxSize: number = 180
    21    @State num: number = 0
    22    scroller: Scroller = new Scroller()
    23    @State colors: number[] = [0xFFC0CB, 0xDA70D6, 0x6B8E23, 0x6A5ACD, 0x00FFFF, 0x00FF7F]
    24    dataSource: WaterFlowDataSource = new WaterFlowDataSource()
    25    @State layoutdir1: FlexDirection = FlexDirection.ColumnReverse
```

### After (local context)
```
    13   * See the License for the specific language governing permissions and
    14   * limitations under the License.
    15   */
    16  import { WaterFlowDataSource } from './WaterFlowDataSource'
    17  @Reusable
    18  @Component
    19  struct ItemTextComponent {
    20    @State item: number = 0
    21  
    22    aboutToReuse(params: Record<string, ESObject>) {
    23      this.item = params.item
    24    }
    25  
```

### Local diff
```diff
--- before/entry/src/main/ets/MainAbility/pages/WaterFlow/waterFlow3.ets
+++ after/entry/src/main/ets/MainAbility/pages/WaterFlow/waterFlow3.ets
@@ -1,3 +1,4 @@
+ *
  *     http://www.apache.org/licenses/LICENSE-2.0
  *
  * Unless required by applicable law or agreed to in writing, software
@@ -7,19 +8,18 @@
  * limitations under the License.
  */
 import { WaterFlowDataSource } from './WaterFlowDataSource'
-@Entry
+@Reusable
 @Component
-struct WaterFlow3 {
-  @State minSize: number = 80
-  @State maxSize: number = 180
-  @State num: number = 0
-  scroller: Scroller = new Scroller()
-  @State colors: number[] = [0xFFC0CB, 0xDA70D6, 0x6B8E23, 0x6A5ACD, 0x00FFFF, 0x00FF7F]
-  dataSource: WaterFlowDataSource = new WaterFlowDataSource()
-  @State layoutdir1: FlexDirection = FlexDirection.ColumnReverse
-  private itemWidthArray: number[] = []
-  private itemHeightArray: number[] = []
+struct ItemTextComponent {
+  @State item: number = 0
 
-  // 计算FlowItem宽/高
-  getSize() {
-    let ret = Math.floor(Math.random() * this.maxSize)
+  aboutToReuse(params: Record<string, ESObject>) {
+    this.item = params.item
+  }
+
+  build() {
+    Text("N" + this.item)
+      .fontSize(12)
+      .height('16')
+  }
+}
```

## 2e3e7bfa2ac4
- project: `ace_ets_module_swiper`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-redundant-state-var` (suggestion)
- file: `entry/src/main/ets/MainAbility/pages/Swiper/Swiper/SwiperVisibility.ets`:44:3
- patch: shape=`modify`, changed_lines=4

### Before (local context)
```
    38  @Entry
    39  @Component
    40  struct SwiperVisibilityTest {
    41    private swiperController: SwiperController = new SwiperController()
    42    private data: MyDataSourceSwiperVisibility = new MyDataSourceSwiperVisibility([])
    43    @State loop: boolean = false
    44    @State index: number = 0
    45    @State autoPlay: boolean = false
    46    @State swiperValue: string = ''
    47    @State text: string = ''
    48  
    49    aboutToAppear(): void {
    50      let list: number[] = []
```

### After (local context)
```
    38  @Entry
    39  @Component
    40  struct SwiperVisibilityTest {
    41    private swiperController: SwiperController = new SwiperController()
    42    private data: MyDataSourceSwiperVisibility = new MyDataSourceSwiperVisibility([])
    43    loop: boolean = false
    44    index: number = 0
    45    autoPlay: boolean = false
    46    @State swiperValue: string = ''
    47    text: string = ''
    48  
    49    aboutToAppear(): void {
    50      let list: number[] = []
```

### Local diff
```diff
--- before/entry/src/main/ets/MainAbility/pages/Swiper/Swiper/SwiperVisibility.ets
+++ after/entry/src/main/ets/MainAbility/pages/Swiper/Swiper/SwiperVisibility.ets
@@ -9,11 +9,11 @@
 struct SwiperVisibilityTest {
   private swiperController: SwiperController = new SwiperController()
   private data: MyDataSourceSwiperVisibility = new MyDataSourceSwiperVisibility([])
-  @State loop: boolean = false
-  @State index: number = 0
-  @State autoPlay: boolean = false
+  loop: boolean = false
+  index: number = 0
+  autoPlay: boolean = false
   @State swiperValue: string = ''
-  @State text: string = ''
+  text: string = ''
 
   aboutToAppear(): void {
     let list: number[] = []
```

## 5939d673e793
- project: `ace_ets_module_swiper`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-unchanged-state-var` (suggestion)
- file: `entry/src/main/ets/MainAbility/pages/Swiper/Swiper07.ets`:46:3
- patch: shape=`modify`, changed_lines=6

### Before (local context)
```
    40  @Component
    41  struct Swiper07 {
    42    private swiperController: SwiperController = new SwiperController();
    43    private data: MyDataSourceSwiper07 = new MyDataSourceSwiper07([]);
    44    @State mywidth: number = 400;
    45    @State myheight: number = 400;
    46    @State displaycount: number = 1;
    47    private arr: number[] = [0, 1, 2, 3, 4, 5, 6];
    48    @State Number: String[] = [
    49      '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15',
    50      '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30'
    51    ];
    52  
```

### After (local context)
```
    40  @Component
    41  struct Swiper07 {
    42    private swiperController: SwiperController = new SwiperController();
    43    private data: MyDataSourceSwiper07 = new MyDataSourceSwiper07([]);
    44    // mywidth 未发生变化，改为普通变量
    45    mywidth: number = 400;
    46    // myheight 未发生变化，改为普通变量
    47    myheight: number = 400;
    48    displaycount: number = 1;
    49    private arr: number[] = [0, 1, 2, 3, 4, 5, 6];
    50    Number: String[] = [
    51      '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15',
    52      '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30'
```

### Local diff
```diff
--- before/entry/src/main/ets/MainAbility/pages/Swiper/Swiper07.ets
+++ after/entry/src/main/ets/MainAbility/pages/Swiper/Swiper07.ets
@@ -8,11 +8,13 @@
 struct Swiper07 {
   private swiperController: SwiperController = new SwiperController();
   private data: MyDataSourceSwiper07 = new MyDataSourceSwiper07([]);
-  @State mywidth: number = 400;
-  @State myheight: number = 400;
-  @State displaycount: number = 1;
+  // mywidth 未发生变化，改为普通变量
+  mywidth: number = 400;
+  // myheight 未发生变化，改为普通变量
+  myheight: number = 400;
+  displaycount: number = 1;
   private arr: number[] = [0, 1, 2, 3, 4, 5, 6];
-  @State Number: String[] = [
+  Number: String[] = [
     '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15',
     '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30'
   ];
@@ -21,5 +23,3 @@
     let list: string[] = []
     for (let i = 1; i <= 10; i++) {
       list.push(i.toString());
-    }
-    this.data = new MyDataSourceSwiper07(list)
```

## a7da34c94637
- project: `ace_ets_module_swiper`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-use-reusable-component` (warn)
- file: `entry/src/main/ets/MainAbility/pages/Swiper/Swiper1.ets`:74:17
- patch: shape=`modify`, changed_lines=66

### Before (local context)
```
    68      Row() {
    69        Column() {
    70          List({ space: 5, initialIndex: 0 }) {
    71            ListItem() {
    72              Swiper(this.swiperController) {
    73                LazyForEach(this.data, (item: string) => {
    74                  Text(item)
    75                    .width('90%')
    76                    .height(160)
    77                    .backgroundColor(0xAFEEEE)
    78                    .textAlign(TextAlign.Center)
    79                    .fontSize(30)
    80                }, (item: string) => item)
```

### After (local context)
```
    68    build() {
    69      Row() {
    70        Column() {
    71          List({ space: 5, initialIndex: 0 }) {
    72            ListItem() {
    73              Swiper(this.swiperController) {
    74                LazyForEach(this.data, (item: string) => {
    75                  Text(item)
    76                    .width('90%')
    77                    .height(160)
    78                    .backgroundColor(0xAFEEEE)
    79                    .textAlign(TextAlign.Center)
    80                    .fontSize(30)
```

### Local diff
```diff
--- before/entry/src/main/ets/MainAbility/pages/Swiper/Swiper1.ets
+++ after/entry/src/main/ets/MainAbility/pages/Swiper/Swiper1.ets
@@ -1,5 +1,6 @@
-  @State selectedWeight: FontWeight | number | string = FontWeight.Normal;
-  @State itemWeight: FontWeight | number | string = FontWeight.Normal;
+  itemSize: string | number = 14;
+  selectedWeight: FontWeight | number | string = FontWeight.Normal;
+  itemWeight: FontWeight | number | string = FontWeight.Normal;
   private swiperController: SwiperController = new SwiperController()
   private data: MyDataSourceTest = new MyDataSourceTest([])
 
@@ -16,10 +17,9 @@
                   .backgroundColor(0xAFEEEE)
                   .textAlign(TextAlign.Center)
                   .fontSize(30)
-              }, (item: string) => item)
+              }, (item: string) => item).cachedCount(4)
             }
             .autoPlay(true)
             .interval(4000)
             .loop(true)
             .duration(1000)
-            .itemSpace(0)
```

## 3487af04f65e
- project: `ace_ets_module_swiper`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-use-reusable-component` (warn)
- file: `entry/src/main/ets/MainAbility/pages/Swiper/Swiper1.ets`:1053:17
- patch: shape=`modify`, changed_lines=66

### Before (local context)
```
  1047              })
  1048            }
  1049  
  1050            ListItem() {
  1051              Swiper(this.swiperController) {
  1052                LazyForEach(this.data, (item: string) => {
  1053                  Text(item)
  1054                    .width('90%')
  1055                    .height(160)
  1056                    .backgroundColor(0xAFEEEE)
  1057                    .textAlign(TextAlign.Center)
  1058                    .fontSize(30)
  1059                }, (item: string) => item)
```

### After (local context)
```
  1047                console.info(index.toString())
  1048              })
  1049            }
  1050  
  1051            ListItem() {
  1052              Swiper(this.swiperController) {
  1053                LazyForEach(this.data, (item: string) => {
  1054                  Text(item)
  1055                    .width('90%')
  1056                    .height(160)
  1057                    .backgroundColor(0xAFEEEE)
  1058                    .textAlign(TextAlign.Center)
  1059                    .fontSize(30)
```

### Local diff
```diff
--- before/entry/src/main/ets/MainAbility/pages/Swiper/Swiper1.ets
+++ after/entry/src/main/ets/MainAbility/pages/Swiper/Swiper1.ets
@@ -1,3 +1,4 @@
+            .borderColor($r("sys.color.ohos_id_color_text_primary"))
             .key("ArkUX_Stage_Swiper_SelectedFontColor_0900")
             .curve(Curve.Linear)
             .indicator(Indicator.digit()
@@ -16,10 +17,9 @@
                   .backgroundColor(0xAFEEEE)
                   .textAlign(TextAlign.Center)
                   .fontSize(30)
-              }, (item: string) => item)
+              }, (item: string) => item).cachedCount(4)
             }
             .autoPlay(true)
             .interval(4000)
             .loop(true)
             .duration(1000)
-            .itemSpace(0)
```

## 5b66c5cab36b
- project: `ace_ets_module_swiper`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-set-cache-count-for-lazyforeach-grid` (suggestion)
- file: `entry/src/main/ets/MainAbility/pages/Swiper/Swiper1_1.ets`:453:13
- patch: shape=`modify`, changed_lines=37

### Before (local context)
```
   447              .onChange((index: number) => {
   448                console.info(index.toString())
   449              })
   450            }
   451  
   452            ListItem() {
   453              Swiper(this.swiperController) {
   454                LazyForEach(this.data, (item: string) => {
   455                  Text(item)
   456                    .width('90%')
   457                    .height(160)
   458                    .backgroundColor(0xAFEEEE)
   459                    .textAlign(TextAlign.Center)
```

### After (local context)
```
   447              .onChange((index: number) => {
   448                console.info(index.toString())
   449              })
   450            }
   451  
   452            ListItem() {
   453              Swiper(this.swiperController) {
   454                LazyForEach(this.data, (item: string) => {
   455                  Text(item)
   456                    .width('90%')
   457                    .height(160)
   458                    .backgroundColor(0xAFEEEE)
   459                    .textAlign(TextAlign.Center)
```

### Local diff
```diff
--- before/entry/src/main/ets/MainAbility/pages/Swiper/Swiper1_1.ets
+++ after/entry/src/main/ets/MainAbility/pages/Swiper/Swiper1_1.ets
@@ -18,7 +18,7 @@
                   .backgroundColor(0xAFEEEE)
                   .textAlign(TextAlign.Center)
                   .fontSize(30)
-              }, (item: string) => item)
+              }, (item: string) => item).cachedCount(4)
             }
             .autoPlay(true)
             .interval(4000)
```

## 161f184108cc
- project: `ace_ets_module_swiper_api11`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-set-cache-count-for-lazyforeach-grid` (suggestion)
- file: `entry/src/main/ets/MainAbility/pages/Swiper/Swiper1.ets`:822:13
- patch: shape=`modify`, changed_lines=65

### Before (local context)
```
   816              .onChange((index: number) => {
   817                console.info(index.toString())
   818              })
   819            }
   820  
   821            ListItem() {
   822              Swiper(this.swiperController) {
   823                LazyForEach(this.data, (item: string) => {
   824                  Text(item)
   825                    .width('90%')
   826                    .height(160)
   827                    .backgroundColor(0xAFEEEE)
   828                    .textAlign(TextAlign.Center)
```

### After (local context)
```
   816              .onChange((index: number) => {
   817                console.info(index.toString())
   818              })
   819            }
   820  
   821            ListItem() {
   822              Swiper(this.swiperController) {
   823                LazyForEach(this.data, (item: string) => {
   824                  Text(item)
   825                    .width('90%')
   826                    .height(160)
   827                    .backgroundColor(0xAFEEEE)
   828                    .textAlign(TextAlign.Center)
```

### Local diff
```diff
--- before/entry/src/main/ets/MainAbility/pages/Swiper/Swiper1.ets
+++ after/entry/src/main/ets/MainAbility/pages/Swiper/Swiper1.ets
@@ -18,7 +18,7 @@
                   .backgroundColor(0xAFEEEE)
                   .textAlign(TextAlign.Center)
                   .fontSize(30)
-              }, (item: string) => item)
+              }, (item: string) => item).cachedCount(4)
             }
             .autoPlay(true)
             .interval(4000)
```

## c1dce6652b98
- project: `ace_ets_module_swiper_api11`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-use-onAnimationStart-for-swiper-preload` (warn)
- file: `entry/src/main/ets/MainAbility/pages/Swiper/Swiper1.ets`:974:13
- patch: shape=`modify`, changed_lines=65

### Before (local context)
```
   968              .onChange((index: number) => {
   969                console.info(index.toString())
   970              })
   971            }
   972  
   973            ListItem() {
   974              Swiper(this.swiperController) {
   975                LazyForEach(this.data, (item: string) => {
   976                  Text(item)
   977                    .width('90%')
   978                    .height(160)
   979                    .backgroundColor(0xAFEEEE)
   980                    .textAlign(TextAlign.Center)
```

### After (local context)
```
   968              .onChange((index: number) => {
   969                console.info(index.toString())
   970              })
   971            }
   972  
   973            ListItem() {
   974              Swiper(this.swiperController) {
   975                LazyForEach(this.data, (item: string) => {
   976                  Text(item)
   977                    .width('90%')
   978                    .height(160)
   979                    .backgroundColor(0xAFEEEE)
   980                    .textAlign(TextAlign.Center)
```

### Local diff
```diff
--- before/entry/src/main/ets/MainAbility/pages/Swiper/Swiper1.ets
+++ after/entry/src/main/ets/MainAbility/pages/Swiper/Swiper1.ets
@@ -18,7 +18,7 @@
                   .backgroundColor(0xAFEEEE)
                   .textAlign(TextAlign.Center)
                   .fontSize(30)
-              }, (item: string) => item)
+              }, (item: string) => item).cachedCount(4)
             }
             .autoPlay(true)
             .interval(4000)
```

## 38042f939c6d
- project: `ace_ets_module_swiper_api11`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-use-onAnimationStart-for-swiper-preload` (warn)
- file: `entry/src/main/ets/MainAbility/pages/Swiper/Swiper1_1.ets`:378:13
- patch: shape=`modify`, changed_lines=395

### Before (local context)
```
   372              .onChange((index: number) => {
   373                console.info(index.toString())
   374              })
   375            }
   376  
   377            ListItem() {
   378              Swiper(this.swiperController) {
   379                LazyForEach(this.data, (item: string) => {
   380                  Text(item)
   381                    .width('90%')
   382                    .height(160)
   383                    .backgroundColor(0xAFEEEE)
   384                    .textAlign(TextAlign.Center)
```

### After (local context)
```
   372              .layoutWeight(null)
   373              .onAnimationStart((index: number, targetIndex: number) => {
   374                if (targetIndex !== index) {
   375                  console.info(targetIndex.toString())
   376                }
   377              })
   378            }
   379  
   380            ListItem() {
   381              Swiper(this.swiperController) {
   382                LazyForEach(this.data, (item: string) => {
   383                  Text(item)
   384                    .width('90%')
```

### Local diff
```diff
--- before/entry/src/main/ets/MainAbility/pages/Swiper/Swiper1_1.ets
+++ after/entry/src/main/ets/MainAbility/pages/Swiper/Swiper1_1.ets
@@ -1,11 +1,14 @@
-            .duration(1000)
             .itemSpace(0)
-            .key('ArkUX_Stage_Swiper_DigitFont_0100')
+            .key('ArkUX_Stage_Swiper_SelectedDigitFont_1100')
             .curve(Curve.Linear)
             .indicator(Indicator.digit()
-              .digitFont({ size: 34, weight: '800' }))
-            .onChange((index: number) => {
-              console.info(index.toString())
+              .selectedDigitFont({ size: 10, weight: 600 }))
+            .size(null)
+            .layoutWeight(null)
+            .onAnimationStart((index: number, targetIndex: number) => {
+              if (targetIndex !== index) {
+                console.info(targetIndex.toString())
+              }
             })
           }
 
@@ -18,8 +21,5 @@
                   .backgroundColor(0xAFEEEE)
                   .textAlign(TextAlign.Center)
                   .fontSize(30)
-              }, (item: string) => item)
+              }, (item: string) => item).cachedCount(4)
             }
-            .autoPlay(true)
-            .interval(4000)
-            .loop(true)
```

## 0bcb3ac10c1e
- project: `ace_ets_module_swiper_api11`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-use-onAnimationStart-for-swiper-preload` (warn)
- file: `entry/src/main/ets/MainAbility/pages/Swiper/Swiper1_1.ets`:478:13
- patch: shape=`modify`, changed_lines=395

### Before (local context)
```
   472              .onChange((index: number) => {
   473                console.info(index.toString())
   474              })
   475            }
   476  
   477            ListItem() {
   478              Swiper(this.swiperController) {
   479                LazyForEach(this.data, (item: string) => {
   480                  Text(item)
   481                    .width('90%')
   482                    .height(160)
   483                    .backgroundColor(0xAFEEEE)
   484                    .textAlign(TextAlign.Center)
```

### After (local context)
```
   472              .autoPlay(true)
   473              .interval(4000)
   474              .loop(true)
   475              .duration(1000)
   476              .itemSpace(0)
   477              .key('ArkUX_Stage_Swiper_DigitFont_0400')
   478              .curve(Curve.Linear)
   479              .indicator(Indicator.digit()
   480                .digitFont({ size: '', weight: FontWeight.Bolder }))
   481              .onAnimationStart((index: number, targetIndex: number) => {
   482                if (targetIndex !== index) {
   483                  console.info(targetIndex.toString())
   484                }
```

### Local diff
```diff
--- before/entry/src/main/ets/MainAbility/pages/Swiper/Swiper1_1.ets
+++ after/entry/src/main/ets/MainAbility/pages/Swiper/Swiper1_1.ets
@@ -1,25 +1,25 @@
+                  .height(160)
+                  .backgroundColor(0xAFEEEE)
+                  .textAlign(TextAlign.Center)
+                  .fontSize(30)
+              }, (item: string) => item).cachedCount(4)
+            }
+            .autoPlay(true)
+            .interval(4000)
+            .loop(true)
             .duration(1000)
             .itemSpace(0)
-            .key('ArkUX_Stage_Swiper_DigitFont_0500')
+            .key('ArkUX_Stage_Swiper_DigitFont_0400')
             .curve(Curve.Linear)
             .indicator(Indicator.digit()
-              .digitFont({ size: -2, weight: 1000 }))
-            .onChange((index: number) => {
-              console.info(index.toString())
+              .digitFont({ size: '', weight: FontWeight.Bolder }))
+            .onAnimationStart((index: number, targetIndex: number) => {
+              if (targetIndex !== index) {
+                console.info(targetIndex.toString())
+              }
             })
           }
 
           ListItem() {
             Swiper(this.swiperController) {
               LazyForEach(this.data, (item: string) => {
-                Text(item)
-                  .width('90%')
-                  .height(160)
-                  .backgroundColor(0xAFEEEE)
-                  .textAlign(TextAlign.Center)
-                  .fontSize(30)
-              }, (item: string) => item)
-            }
-            .autoPlay(true)
-            .interval(4000)
-            .loop(true)
```

## 79b10a0dae55
- project: `ace_ets_module_swiper_api11`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-use-reusable-component` (warn)
- file: `entry/src/main/ets/MainAbility/pages/Swiper/swiperNew.ets`:71:13
- patch: shape=`modify`, changed_lines=28

### Before (local context)
```
    65      Flex({ direction: FlexDirection.Column, alignItems: ItemAlign.Center, justifyContent: FlexAlign.Center }) {
    66  
    67        Column({ space: 5 }) {
    68  
    69          Swiper(this.swiperController) {
    70            LazyForEach(this.data, (item: string) => {
    71              Text(item)
    72                .width('90%')
    73                .height(160)
    74                .backgroundColor(0xAFEEEE)
    75                .textAlign(TextAlign.Center)
    76                .fontSize(20)
    77            }, (item: string) => item)
```

### After (local context)
```
    65  @Entry
    66  @Component
    67  struct SwiperCurve {
    68    private swiperController: SwiperController = new SwiperController()
    69    private data: MyDataSource = new MyDataSource([])
    70  
    71    aboutToAppear(): void {
    72      Log.showInfo(TAG, `aboutToAppear SwiperCurve start`)
    73      let list: number[] = []
    74      for (let i = 1; i <= 10; i++) {
    75        list.push(i);
    76      }
    77      this.data = new MyDataSource(list)
```

### Local diff
```diff
--- before/entry/src/main/ets/MainAbility/pages/Swiper/swiperNew.ets
+++ after/entry/src/main/ets/MainAbility/pages/Swiper/swiperNew.ets
@@ -1,25 +1,25 @@
+  }
+}
+
+
+const TAG = 'ets_apiLack_add';
+
+@Entry
+@Component
+struct SwiperCurve {
+  private swiperController: SwiperController = new SwiperController()
+  private data: MyDataSource = new MyDataSource([])
+
+  aboutToAppear(): void {
+    Log.showInfo(TAG, `aboutToAppear SwiperCurve start`)
+    let list: number[] = []
+    for (let i = 1; i <= 10; i++) {
+      list.push(i);
+    }
+    this.data = new MyDataSource(list)
+  }
 
   aboutToDisappear() {
     Log.showInfo(TAG, `aboutToDisAppear SwiperCurve end`)
   }
 
-  build() {
-    Flex({ direction: FlexDirection.Column, alignItems: ItemAlign.Center, justifyContent: FlexAlign.Center }) {
-
-      Column({ space: 5 }) {
-
-        Swiper(this.swiperController) {
-          LazyForEach(this.data, (item: string) => {
-            Text(item)
-              .width('90%')
-              .height(160)
-              .backgroundColor(0xAFEEEE)
-              .textAlign(TextAlign.Center)
-              .fontSize(20)
-          }, (item: string) => item)
-        }
-        .key("swiper")
-        .cachedCount(2)
-        .index(1)
-        .autoPlay(true)
-        .interval(4000)
```

## b789c97bf7c1
- project: `ace_ets_module_swiper_api11`
- round_fixed: `2` (early)
- rule: `performance/hp-arkui-use-onAnimationStart-for-swiper-preload` (warn)
- file: `entry/src/main/ets/MainAbility/pages/Swiper/swiperUI/swiperNextPrevMargin54.ets`:55:7
- patch: shape=`modify`, changed_lines=29

### Before (local context)
```
    49      }
    50      this.data = new MyDataSource54(list)
    51    }
    52  
    53    build() {
    54      Column({ space: 5 }) {
    55        Swiper(this.swiperController) {
    56          LazyForEach(this.data, (item: string,index:number) => {
    57            Text(item)
    58              .width('100%')
    59              .height('100%')
    60              .backgroundColor(0xAFEEEE)
    61              .textAlign(TextAlign.Center)
```

### After (local context)
```
    49      Text(this.item)
    50        .width('100%')
    51        .height('100%')
    52        .backgroundColor(0xAFEEEE)
    53        .textAlign(TextAlign.Center)
    54        .borderColor(Color.Red)
    55        .borderWidth(0)
    56        .fontSize(30)
    57    }
    58  }
    59  
    60  @Entry
    61  @Component
```

### Local diff
```diff
--- before/entry/src/main/ets/MainAbility/pages/Swiper/swiperUI/swiperNextPrevMargin54.ets
+++ after/entry/src/main/ets/MainAbility/pages/Swiper/swiperUI/swiperNextPrevMargin54.ets
@@ -1,25 +1,25 @@
+
+  aboutToReuse(params: Record<string, ESObject>) {
+    this.item = params.item
+  }
+
+  build() {
+    Text(this.item)
+      .width('100%')
+      .height('100%')
+      .backgroundColor(0xAFEEEE)
+      .textAlign(TextAlign.Center)
+      .borderColor(Color.Red)
+      .borderWidth(0)
+      .fontSize(30)
+  }
+}
+
+@Entry
+@Component
+struct SwiperNextPrevMarginExample54 {
+  private swiperController: SwiperController = new SwiperController()
   private data: MyDataSource54 = new MyDataSource54([])
 
   aboutToAppear(): void {
     let list: number[] = []
-    for (let i = 1; i <= 10; i++) {
-      list.push(i);
-    }
-    this.data = new MyDataSource54(list)
-  }
-
-  build() {
-    Column({ space: 5 }) {
-      Swiper(this.swiperController) {
-        LazyForEach(this.data, (item: string,index:number) => {
-          Text(item)
-            .width('100%')
-            .height('100%')
-            .backgroundColor(0xAFEEEE)
-            .textAlign(TextAlign.Center)
-            .borderColor(Color.Red)
-            .borderWidth(0)
-            .fontSize(30).key(index+"")
-        }, (item: string) => item)
-      }
-      .cachedCount(2)
```

## 5ea380461e57
- project: `acts_validator`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-avoid-empty-callback` (suggestion)
- file: `entry/src/main/ets/pages/ArkUI/CanvasFont.ets`:511:11
- patch: shape=`modify`, changed_lines=474

### Before (local context)
```
   505  
   506        Flex({ direction: FlexDirection.Column, alignItems: ItemAlign.Center, justifyContent: FlexAlign.Center }) {
   507          Canvas(this.context)
   508            .width('100%')
   509            .height('100%')
   510            .backgroundColor('#ffffffff')
   511            .onReady(() => {
   512            })
   513        }
   514        .width('100%')
   515        .height('40%')
   516      }
   517      .width('100%')
```

### After (local context)
```
   505        .height('60%')
   506  
   507        Flex({ direction: FlexDirection.Column, alignItems: ItemAlign.Center, justifyContent: FlexAlign.Center }) {
   508          Canvas(this.context)
   509            .width('100%')
   510            .height('100%')
   511            .backgroundColor('#ffffffff')
   512            .onReady(() => {
   513             this.handleScrollEvent()
   514            })
   515        }
   516        .width('100%')
   517        .height('40%')
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/ArkUI/CanvasFont.ets
+++ after/entry/src/main/ets/pages/ArkUI/CanvasFont.ets
@@ -1,5 +1,6 @@
-          .height('200vp')
-        }
+        .width('90%')
+        .height('200vp')
+        
       }
       .scrollBarWidth('6vp')
       .scrollBarColor('#cccccc')
@@ -11,6 +12,7 @@
           .height('100%')
           .backgroundColor('#ffffffff')
           .onReady(() => {
+           this.handleScrollEvent()
           })
       }
       .width('100%')
@@ -21,5 +23,3 @@
     .backgroundColor(Color.White)
   }
 
-  build() {
-    Column() {
```

## 0b743e76c53a
- project: `acts_validator`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-unchanged-state-var` (suggestion)
- file: `entry/src/main/ets/pages/ArkUI/CapiHandDetect.ets`:78:3
- patch: shape=`modify`, changed_lines=17

### Before (local context)
```
    72      '操作步骤：左手拿手机，左手长按+拖动‘顺序手势组’按钮\n' + '预期结果：旗舰手机hand1与hand2打印1，其它设备均打印0“\n' +
    73      '操作步骤：左手拿手机，右手长按+拖动‘顺序手势组’按钮\n' + '预期结果：旗舰手机hand1与hand2打印2，其它设备均打印0“\n' +
    74      '操作步骤：右手拿手机，左手长按+拖动‘顺序手势组’按钮\n' + '预期结果：旗舰手机hand1与hand2打印1，其它设备均打印0“\n' +
    75      '操作步骤：右手拿手机，右手长按+拖动‘顺序手势组’按钮\n' + '预期结果：旗舰手机hand1与hand2打印2，其它设备均打印0“\n' +
    76      '操作步骤：双指长按+拖动‘顺序手势组’按钮\n' + '预期结果：hand1与hand2打印0“\n';
    77    @State CapiHandDetectVue: boolean = false;
    78    @State active: boolean = false;
    79    @State intervalNum: number = 0;
    80    @State yesEnable: boolean = false
    81    // 默认不支持
    82    @State onAccessibilityHoverEnable: boolean = false;
    83    @State hoverText: string = 'no hover';
    84    @State color: Color = Color.Blue;
```

### After (local context)
```
    72      '操作步骤：右手拿手机，左手触摸‘单一手势+onTouch’按钮\n' + '预期结果：旗舰手机hand1与hand2打印1，其它设备均打印0“\n' +
    73      '操作步骤：右手拿手机，右手触摸‘单一手势+onTouch’按钮\n' + '预期结果：旗舰手机hand1与hand2打印2，其它设备均打印0“\n' +
    74      '操作步骤：双指触摸‘单一手势+onTouch’按钮\n' + '预期结果：hand1与hand2打印0“\n' +
    75      '操作步骤：左手拿手机，左手长按+拖动‘顺序手势组’按钮\n' + '预期结果：旗舰手机hand1与hand2打印1，其它设备均打印0“\n' +
    76      '操作步骤：左手拿手机，右手长按+拖动‘顺序手势组’按钮\n' + '预期结果：旗舰手机hand1与hand2打印2，其它设备均打印0“\n' +
    77      '操作步骤：右手拿手机，左手长按+拖动‘顺序手势组’按钮\n' + '预期结果：旗舰手机hand1与hand2打印1，其它设备均打印0“\n' +
    78      '操作步骤：右手拿手机，右手长按+拖动‘顺序手势组’按钮\n' + '预期结果：旗舰手机hand1与hand2打印2，其它设备均打印0“\n' +
    79      '操作步骤：双指长按+拖动‘顺序手势组’按钮\n' + '预期结果：hand1与hand2打印0“\n';
    80    @State CapiHandDetectVue: boolean = false;
    81    active: boolean = false;
    82    intervalNum: number = 0;
    83    yesEnable: boolean = false
    84    // 默认不支持
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/ArkUI/CapiHandDetect.ets
+++ after/entry/src/main/ets/pages/ArkUI/CapiHandDetect.ets
@@ -1,3 +1,6 @@
+    '操作步骤：左手拿手机，右手长按+拖动‘并行手势组’按钮\n' + '预期结果：旗舰手机hand1与hand2打印2，其它设备均打印0“\n' +
+    '操作步骤：右手拿手机，左手长按+拖动‘并行手势组’按钮\n' + '预期结果：旗舰手机hand1与hand2打印1，其它设备均打印0“\n' +
+    '操作步骤：右手拿手机，右手长按+拖动‘并行手势组’按钮\n' + '预期结果：旗舰手机hand1与hand2打印2，其它设备均打印0“\n' +
     '操作步骤：双指长按+拖动‘并行手势组’按钮\n' + '预期结果：hand1与hand2打印0“\n' +
     '操作步骤：左手拿手机，左手触摸‘单一手势+onTouch’按钮\n' + '预期结果：旗舰手机hand1与hand2打印1，其它设备均打印0“\n' +
     '操作步骤：左手拿手机，右手触摸‘单一手势+onTouch’按钮\n' + '预期结果：旗舰手机hand1与hand2打印2，其它设备均打印0“\n' +
@@ -10,16 +13,13 @@
     '操作步骤：右手拿手机，右手长按+拖动‘顺序手势组’按钮\n' + '预期结果：旗舰手机hand1与hand2打印2，其它设备均打印0“\n' +
     '操作步骤：双指长按+拖动‘顺序手势组’按钮\n' + '预期结果：hand1与hand2打印0“\n';
   @State CapiHandDetectVue: boolean = false;
-  @State active: boolean = false;
-  @State intervalNum: number = 0;
-  @State yesEnable: boolean = false
+  active: boolean = false;
+  intervalNum: number = 0;
+  yesEnable: boolean = false
   // 默认不支持
-  @State onAccessibilityHoverEnable: boolean = false;
-  @State hoverText: string = 'no hover';
-  @State color: Color = Color.Blue;
-  @State text1: string = ""
-  @State text2: string = ""
-  @State text3: string = ""
-  @State text4: string = ""
-
-  @Builder
+  onAccessibilityHoverEnable: boolean = false;
+  hoverText: string = 'no hover';
+  color: Color = Color.Blue;
+  text1: string = ""
+  text2: string = ""
+  text3: string = ""
```

## ac536d17f87b
- project: `acts_validator`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-redundant-state-var` (suggestion)
- file: `entry/src/main/ets/pages/ArkUI/HoverSpen.ets`:25:1
- patch: shape=`modify`, changed_lines=13

### Before (local context)
```
    19  @Component
    20  struct HoverSpenExample {
    21  @State text: string = ''
    22  @State axisValue: string = ''
    23  @State name: string = 'HoverSpen';
    24  @State Vue: boolean = false;
    25  @State intervalNum: number = 0;
    26  @State hoverMoveText: string = '';
    27  @State StepTips: string = '平板设备使用手写笔测试时\n' + '操作步骤:\n' + '1,手写笔悬浮在蓝色按钮上并且移动\n' +
    28  '预期结果：\n' + '1.页面上出现x轴,y轴坐标(手写笔位置相对于当前组件左上角的x轴坐标，y轴坐标)\n' +
    29  '2.页面上出现windowXY(手写笔位置相对于应用窗口左上角的x轴坐标,y轴坐标)\n' +
    30  '3.页面上出现displayXY(手写笔位置相对于屏幕左上角的x轴坐标,y轴坐标)\n'+
    31  '4.随着手写笔的移动,坐标在不断的变化\n' +
```

### After (local context)
```
    19  @Component
    20  struct HoverSpenExample {
    21  // text 未发生变化，改为普通变量
    22  text: string = ''
    23  // axisValue 未发生变化，改为普通变量
    24  axisValue: string = ''
    25  // name 未发生变化，改为普通变量
    26  name: string = 'HoverSpen';
    27  @State Vue: boolean = false;
    28  intervalNum: number = 0;
    29  @State hoverMoveText: string = '';
    30  // StepTips 未发生变化，改为普通变量
    31  StepTips: string = '平板设备使用手写笔测试时\n' + '操作步骤:\n' + '1,手写笔悬浮在蓝色按钮上并且移动\n' +
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/ArkUI/HoverSpen.ets
+++ after/entry/src/main/ets/pages/ArkUI/HoverSpen.ets
@@ -6,20 +6,20 @@
 @Entry
 @Component
 struct HoverSpenExample {
-@State text: string = ''
-@State axisValue: string = ''
-@State name: string = 'HoverSpen';
+// text 未发生变化，改为普通变量
+text: string = ''
+// axisValue 未发生变化，改为普通变量
+axisValue: string = ''
+// name 未发生变化，改为普通变量
+name: string = 'HoverSpen';
 @State Vue: boolean = false;
-@State intervalNum: number = 0;
+intervalNum: number = 0;
 @State hoverMoveText: string = '';
-@State StepTips: string = '平板设备使用手写笔测试时\n' + '操作步骤:\n' + '1,手写笔悬浮在蓝色按钮上并且移动\n' +
+// StepTips 未发生变化，改为普通变量
+StepTips: string = '平板设备使用手写笔测试时\n' + '操作步骤:\n' + '1,手写笔悬浮在蓝色按钮上并且移动\n' +
 '预期结果：\n' + '1.页面上出现x轴,y轴坐标(手写笔位置相对于当前组件左上角的x轴坐标，y轴坐标)\n' +
 '2.页面上出现windowXY(手写笔位置相对于应用窗口左上角的x轴坐标,y轴坐标)\n' +
 '3.页面上出现displayXY(手写笔位置相对于屏幕左上角的x轴坐标,y轴坐标)\n'+
 '4.随着手写笔的移动,坐标在不断的变化\n' +
 '5.不支持手写笔的设备直接按照失败处理走豁免\n'
 
-aboutToAppear(): void {
-
-FirstDialog.ChooseDialog(this.StepTips, this.name);
-}
```

## 7d16575dbd50
- project: `acts_validator`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-redundant-state-var` (suggestion)
- file: `entry/src/main/ets/pages/ArkUI/PatternLockL0_02.ets`:33:3
- patch: shape=`modify`, changed_lines=7

### Before (local context)
```
    27  struct PatternLockL0_02 {
    28    @State rating: number = 0;
    29    @State FillColor: string = '#FF000000';
    30    @State name: string = 'PatternLockL0_02';
    31    @State StepTips: string = '操作步骤：在两个九宫格中画出自己想设置的密码，点击确认按钮' + '\n' + '预期结果：背景圆环及连线在宫格圆点上方显示';
    32    @State Vue: boolean = false;
    33    @State passwords: Number[] = [];
    34    @State customModifier: PatternLockModifier = new PatternLockModifier()
    35      .sideLength('47.5%')
    36      .circleRadius(9)
    37      .pathStrokeWidth(18)
    38      .regularColor('#ff182431')
    39      .activeColor('#B0C4DE')
```

### After (local context)
```
    27  @Component
    28  struct PatternLockL0_02 {
    29    rating: number = 0;
    30    FillColor: string = '#FF000000';
    31    name: string = 'PatternLockL0_02';
    32    StepTips: string = '操作步骤：在两个九宫格中画出自己想设置的密码，点击确认按钮' + '\n' + '预期结果：背景圆环及连线在宫格圆点上方显示';
    33    @State Vue: boolean = false;
    34    passwords: Number[] = [];
    35    @State customModifier: PatternLockModifier = new PatternLockModifier()
    36      .sideLength('47.5%')
    37      .circleRadius(9)
    38      .pathStrokeWidth(18)
    39      .regularColor('#ff182431')
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/ArkUI/PatternLockL0_02.ets
+++ after/entry/src/main/ets/pages/ArkUI/PatternLockL0_02.ets
@@ -1,3 +1,4 @@
+import { BusinessError } from '@ohos.base';
 import { LengthMetrics, PatternLockModifier } from '@kit.ArkUI';
 
 const TAG = '[PatternLockL0_02]';
@@ -5,12 +6,12 @@
 @Entry
 @Component
 struct PatternLockL0_02 {
-  @State rating: number = 0;
-  @State FillColor: string = '#FF000000';
-  @State name: string = 'PatternLockL0_02';
-  @State StepTips: string = '操作步骤：在两个九宫格中画出自己想设置的密码，点击确认按钮' + '\n' + '预期结果：背景圆环及连线在宫格圆点上方显示';
+  rating: number = 0;
+  FillColor: string = '#FF000000';
+  name: string = 'PatternLockL0_02';
+  StepTips: string = '操作步骤：在两个九宫格中画出自己想设置的密码，点击确认按钮' + '\n' + '预期结果：背景圆环及连线在宫格圆点上方显示';
   @State Vue: boolean = false;
-  @State passwords: Number[] = [];
+  passwords: Number[] = [];
   @State customModifier: PatternLockModifier = new PatternLockModifier()
     .sideLength('47.5%')
     .circleRadius(9)
@@ -22,4 +23,3 @@
     .backgroundColor('#F5F5F5')
     .autoReset(true)
     .activateCircleStyle({
-      radius: LengthMetrics.vp(8),
```

## 3aadad9b49f1
- project: `acts_validator`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-unchanged-state-var` (suggestion)
- file: `entry/src/main/ets/pages/ArkUI/PinchGestureListener.ets`:36:3
- patch: shape=`modify`, changed_lines=9

### Before (local context)
```
    30  
    31  @Entry
    32  @Component
    33  struct PinchGestureExample {
    34    @State value: string = ''
    35    @State name: string = 'PinchGestureListener';
    36    @State scaleValue: number = 1
    37    @State pinchValue: number = 1
    38    @State pinchX: number = 0
    39    @State pinchY: number = 0
    40    @State stepTips: string = '操作步骤:\n' + '1.点击[点击添加监听]按钮\n' +
    41      '2.双指捏合Column组件\n' + '3.过滤日志[aaa ---]\n' +
    42      '4.点击[点击移除监听]按钮\n' + '5.双指捏合Column组件\n' + '6.过滤日志[aaa ---]\n' +
```

### After (local context)
```
    30  }
    31  
    32  @Entry
    33  @Component
    34  struct PinchGestureExample {
    35    value: string = ''
    36    name: string = 'PinchGestureListener';
    37    scaleValue: number = 1
    38    @State pinchValue: number = 1
    39    pinchX: number = 0
    40    pinchY: number = 0
    41    stepTips: string = '操作步骤:\n' + '1.点击[点击添加监听]按钮\n' +
    42      '2.双指捏合Column组件\n' + '3.过滤日志[aaa ---]\n' +
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/ArkUI/PinchGestureListener.ets
+++ after/entry/src/main/ets/pages/ArkUI/PinchGestureListener.ets
@@ -1,3 +1,4 @@
+
 function pinchGesture(info: GestureTriggerInfo) {
   console.info('aaa --- callback event' + JSON.stringify(info.event))
   console.info('aaa --- callback current getTag:' + JSON.stringify(info.current))
@@ -8,18 +9,17 @@
 @Entry
 @Component
 struct PinchGestureExample {
-  @State value: string = ''
-  @State name: string = 'PinchGestureListener';
-  @State scaleValue: number = 1
+  value: string = ''
+  name: string = 'PinchGestureListener';
+  scaleValue: number = 1
   @State pinchValue: number = 1
-  @State pinchX: number = 0
-  @State pinchY: number = 0
-  @State stepTips: string = '操作步骤:\n' + '1.点击[点击添加监听]按钮\n' +
+  pinchX: number = 0
+  pinchY: number = 0
+  stepTips: string = '操作步骤:\n' + '1.点击[点击添加监听]按钮\n' +
     '2.双指捏合Column组件\n' + '3.过滤日志[aaa ---]\n' +
     '4.点击[点击移除监听]按钮\n' + '5.双指捏合Column组件\n' + '6.过滤日志[aaa ---]\n' +
     '预期结果：\n' + '1.添加监听后callback回调日志能够触发\n' +
     '2.添加监听后先触发callback回调日志，再触发onAction start事件日志，再触发callback回调日志，最后触发onAction end事件日志\n' +
     '3.移除监听后callback回调函数不会触发，只会触发onAction start/end事件的日志'
-  @State isOk: boolean = false;
-  @State intervalNum: number = 0;
-
+  isOk: boolean = false;
+  intervalNum: number = 0;
```

## e492f314d219
- project: `acts_validator`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-unchanged-state-var` (suggestion)
- file: `entry/src/main/ets/pages/ArkUI/Progress.ets`:28:3
- patch: shape=`modify`, changed_lines=8

### Before (local context)
```
    22  
    23  const TAG = '[Progress]';
    24  
    25  @Entry
    26  @Component
    27  struct progress {
    28    @State FillColor: string = '#FF000000';
    29    @State name: string = 'Progress';
    30    @State StepTips: string = '操作步骤：拖动滑块向右移动' + '\n' + '预期结果：进度条进度随着拖动距离增大而变大';
    31    @State Vue: boolean = false;
    32    @State accumulateTime: number = 0;
    33    @State progress: number = 0;
    34    @State progressNext: number = 0;
```

### After (local context)
```
    22  
    23  const TAG = '[Progress]';
    24  
    25  @Entry
    26  @Component
    27  struct progress {
    28    FillColor: string = '#FF000000';
    29    name: string = 'Progress';
    30    StepTips: string = '操作步骤：拖动滑块向右移动' + '\n' + '预期结果：进度条进度随着拖动距离增大而变大';
    31    @State Vue: boolean = false;
    32    accumulateTime: number = 0;
    33    @State progress: number = 0;
    34    progressNext: number = 0;
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/ArkUI/Progress.ets
+++ after/entry/src/main/ets/pages/ArkUI/Progress.ets
@@ -10,14 +10,14 @@
 @Entry
 @Component
 struct progress {
-  @State FillColor: string = '#FF000000';
-  @State name: string = 'Progress';
-  @State StepTips: string = '操作步骤：拖动滑块向右移动' + '\n' + '预期结果：进度条进度随着拖动距离增大而变大';
+  FillColor: string = '#FF000000';
+  name: string = 'Progress';
+  StepTips: string = '操作步骤：拖动滑块向右移动' + '\n' + '预期结果：进度条进度随着拖动距离增大而变大';
   @State Vue: boolean = false;
-  @State accumulateTime: number = 0;
+  accumulateTime: number = 0;
   @State progress: number = 0;
-  @State progressNext: number = 0;
-  @State intervalNum: number = 0;
+  progressNext: number = 0;
+  intervalNum: number = 0;
 
   @Builder
   PassBtn(text: Resource, isFullScreen: boolean) {
```

## 350fc1b05534
- project: `acts_validator`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-unchanged-state-var` (suggestion)
- file: `entry/src/main/ets/pages/ArkUI/Progress.ets`:30:3
- patch: shape=`modify`, changed_lines=8

### Before (local context)
```
    24  
    25  @Entry
    26  @Component
    27  struct progress {
    28    @State FillColor: string = '#FF000000';
    29    @State name: string = 'Progress';
    30    @State StepTips: string = '操作步骤：拖动滑块向右移动' + '\n' + '预期结果：进度条进度随着拖动距离增大而变大';
    31    @State Vue: boolean = false;
    32    @State accumulateTime: number = 0;
    33    @State progress: number = 0;
    34    @State progressNext: number = 0;
    35    @State intervalNum: number = 0;
    36  
```

### After (local context)
```
    24  
    25  @Entry
    26  @Component
    27  struct progress {
    28    FillColor: string = '#FF000000';
    29    name: string = 'Progress';
    30    StepTips: string = '操作步骤：拖动滑块向右移动' + '\n' + '预期结果：进度条进度随着拖动距离增大而变大';
    31    @State Vue: boolean = false;
    32    accumulateTime: number = 0;
    33    @State progress: number = 0;
    34    progressNext: number = 0;
    35    intervalNum: number = 0;
    36  
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/ArkUI/Progress.ets
+++ after/entry/src/main/ets/pages/ArkUI/Progress.ets
@@ -8,14 +8,14 @@
 @Entry
 @Component
 struct progress {
-  @State FillColor: string = '#FF000000';
-  @State name: string = 'Progress';
-  @State StepTips: string = '操作步骤：拖动滑块向右移动' + '\n' + '预期结果：进度条进度随着拖动距离增大而变大';
+  FillColor: string = '#FF000000';
+  name: string = 'Progress';
+  StepTips: string = '操作步骤：拖动滑块向右移动' + '\n' + '预期结果：进度条进度随着拖动距离增大而变大';
   @State Vue: boolean = false;
-  @State accumulateTime: number = 0;
+  accumulateTime: number = 0;
   @State progress: number = 0;
-  @State progressNext: number = 0;
-  @State intervalNum: number = 0;
+  progressNext: number = 0;
+  intervalNum: number = 0;
 
   @Builder
   PassBtn(text: Resource, isFullScreen: boolean) {
```

## 159751a6dc9c
- project: `acts_validator`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-unchanged-state-var` (suggestion)
- file: `entry/src/main/ets/pages/ArkUI/Swiper.ets`:32:3
- patch: shape=`modify`, changed_lines=26

### Before (local context)
```
    26  @Component
    27  struct swiper {
    28    @State rating: number = 0;
    29    @State address: string = '';
    30    @State FillColor: string = '#FF000000';
    31    @State name: string = 'Swiper';
    32    @State StepTips: string = '操作步骤：点击上一页或下一页按钮，可以正常翻页' + '\n' + '预期结果：轮播图能够正常翻页';
    33    @State Vue: boolean = false;
    34    @State active: boolean = false;
    35    @State intervalNum: number = 0;
    36    @State isDisableSwipe: boolean = false;
    37    @State itemSpace: number = 0;
    38    @State isVertical: boolean = false;
```

### After (local context)
```
    26  @Entry
    27  @Component
    28  struct swiper {
    29    rating: number = 0;
    30    address: string = '';
    31    FillColor: string = '#FF000000';
    32    // name 未发生变化，改为普通变量
    33    name: string = 'Swiper';
    34    // StepTips 未发生变化，改为普通变量
    35    StepTips: string = '操作步骤：点击上一页或下一页按钮，可以正常翻页' + '\n' + '预期结果：轮播图能够正常翻页';
    36    @State Vue: boolean = false;
    37    active: boolean = false;
    38    intervalNum: number = 0;
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/ArkUI/Swiper.ets
+++ after/entry/src/main/ets/pages/ArkUI/Swiper.ets
@@ -1,3 +1,4 @@
+import fs from '@ohos.file.fs';
 import FirstDialog from '../model/FirstDialog';
 import { BusinessError } from '@ohos.base';
 
@@ -6,20 +7,19 @@
 @Entry
 @Component
 struct swiper {
-  @State rating: number = 0;
-  @State address: string = '';
-  @State FillColor: string = '#FF000000';
-  @State name: string = 'Swiper';
-  @State StepTips: string = '操作步骤：点击上一页或下一页按钮，可以正常翻页' + '\n' + '预期结果：轮播图能够正常翻页';
+  rating: number = 0;
+  address: string = '';
+  FillColor: string = '#FF000000';
+  // name 未发生变化，改为普通变量
+  name: string = 'Swiper';
+  // StepTips 未发生变化，改为普通变量
+  StepTips: string = '操作步骤：点击上一页或下一页按钮，可以正常翻页' + '\n' + '预期结果：轮播图能够正常翻页';
   @State Vue: boolean = false;
-  @State active: boolean = false;
-  @State intervalNum: number = 0;
-  @State isDisableSwipe: boolean = false;
-  @State itemSpace: number = 0;
-  @State isVertical: boolean = false;
-  @State duration: number = 400;
-  @State loop: boolean = true;
-  @State autoPlay: boolean = false;
-  @State interval: number = 1000;
-  @State count: number = 0;
-  private controller: SwiperController = new SwiperController();
+  active: boolean = false;
+  intervalNum: number = 0;
+  // isDisableSwipe 未发生变化，改为普通变量
+  isDisableSwipe: boolean = false;
+  // itemSpace 未发生变化，改为普通变量
+  itemSpace: number = 0;
+  // isVertical 未发生变化，改为普通变量
+  isVertical: boolean = false;
```

## 1cb470c80a7e
- project: `acts_validator`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-redundant-state-var` (suggestion)
- file: `entry/src/main/ets/pages/ArkUI/enableDropDisallowedBadge.ets`:24:3
- patch: shape=`modify`, changed_lines=15

### Before (local context)
```
    18  
    19  const TAG = '[enableDropDisallowedBadge]';
    20  
    21  @Entry
    22  @Component
    23  struct enableDropDisallowedBadgeExample {
    24    @State text: string = ''
    25    @State axisValue: string = ''
    26    @State name: string = 'enableDropDisallowedBadge';
    27    @State Vue: boolean = false;
    28    @State intervalNum: number = 0;
    29    @State hoverMoveText: string = '';
    30    @State StepTips: string = '操作步骤：将图片拖至禁止角标区域' + '\n' + '预期结果:显示禁止标识,结果为True'
```

### After (local context)
```
    18  import { unifiedDataChannel, uniformTypeDescriptor } from '@kit.ArkData';
    19  
    20  const TAG = '[enableDropDisallowedBadge]';
    21  
    22  @Entry
    23  @Component
    24  struct enableDropDisallowedBadgeExample {
    25    text: string = ''
    26    axisValue: string = ''
    27    name: string = 'enableDropDisallowedBadge';
    28    @State Vue: boolean = false;
    29    intervalNum: number = 0;
    30    hoverMoveText: string = '';
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/ArkUI/enableDropDisallowedBadge.ets
+++ after/entry/src/main/ets/pages/ArkUI/enableDropDisallowedBadge.ets
@@ -1,3 +1,4 @@
+ See the License for the specific language governing permissions and
  limitations under the License.
  */
 import router from '@ohos.router';
@@ -10,16 +11,15 @@
 @Entry
 @Component
 struct enableDropDisallowedBadgeExample {
-  @State text: string = ''
-  @State axisValue: string = ''
-  @State name: string = 'enableDropDisallowedBadge';
+  text: string = ''
+  axisValue: string = ''
+  name: string = 'enableDropDisallowedBadge';
   @State Vue: boolean = false;
-  @State intervalNum: number = 0;
-  @State hoverMoveText: string = '';
-  @State StepTips: string = '操作步骤：将图片拖至禁止角标区域' + '\n' + '预期结果:显示禁止标识,结果为True'
-  @State uri: string = ""
-  @State AblockArr: string[] = []
-  @State BblockArr: string[] = []
-  @State AVisible: Visibility = Visibility.Visible
+  intervalNum: number = 0;
+  hoverMoveText: string = '';
+  StepTips: string = '操作步骤：将图片拖至禁止角标区域' + '\n' + '预期结果:显示禁止标识,结果为True'
+  uri: string = ""
+  AblockArr: string[] = []
+  BblockArr: string[] = []
+  AVisible: Visibility = Visibility.Visible
   @State dragSuccess: Boolean = false
-
```

## 961791a96d63
- project: `acts_validator`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-unchanged-state-var` (suggestion)
- file: `entry/src/main/ets/pages/ArkUI/enableDropDisallowedBadge.ets`:29:3
- patch: shape=`modify`, changed_lines=15

### Before (local context)
```
    23  struct enableDropDisallowedBadgeExample {
    24    @State text: string = ''
    25    @State axisValue: string = ''
    26    @State name: string = 'enableDropDisallowedBadge';
    27    @State Vue: boolean = false;
    28    @State intervalNum: number = 0;
    29    @State hoverMoveText: string = '';
    30    @State StepTips: string = '操作步骤：将图片拖至禁止角标区域' + '\n' + '预期结果:显示禁止标识,结果为True'
    31    @State uri: string = ""
    32    @State AblockArr: string[] = []
    33    @State BblockArr: string[] = []
    34    @State AVisible: Visibility = Visibility.Visible
    35    @State dragSuccess: Boolean = false
```

### After (local context)
```
    23  @Component
    24  struct enableDropDisallowedBadgeExample {
    25    text: string = ''
    26    axisValue: string = ''
    27    name: string = 'enableDropDisallowedBadge';
    28    @State Vue: boolean = false;
    29    intervalNum: number = 0;
    30    hoverMoveText: string = '';
    31    StepTips: string = '操作步骤：将图片拖至禁止角标区域' + '\n' + '预期结果:显示禁止标识,结果为True'
    32    uri: string = ""
    33    AblockArr: string[] = []
    34    BblockArr: string[] = []
    35    AVisible: Visibility = Visibility.Visible
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/ArkUI/enableDropDisallowedBadge.ets
+++ after/entry/src/main/ets/pages/ArkUI/enableDropDisallowedBadge.ets
@@ -1,3 +1,4 @@
+import FirstDialog from '../model/FirstDialog';
 import { unifiedDataChannel, uniformTypeDescriptor } from '@kit.ArkData';
 
 const TAG = '[enableDropDisallowedBadge]';
@@ -5,21 +6,20 @@
 @Entry
 @Component
 struct enableDropDisallowedBadgeExample {
-  @State text: string = ''
-  @State axisValue: string = ''
-  @State name: string = 'enableDropDisallowedBadge';
+  text: string = ''
+  axisValue: string = ''
+  name: string = 'enableDropDisallowedBadge';
   @State Vue: boolean = false;
-  @State intervalNum: number = 0;
-  @State hoverMoveText: string = '';
-  @State StepTips: string = '操作步骤：将图片拖至禁止角标区域' + '\n' + '预期结果:显示禁止标识,结果为True'
-  @State uri: string = ""
-  @State AblockArr: string[] = []
-  @State BblockArr: string[] = []
-  @State AVisible: Visibility = Visibility.Visible
+  intervalNum: number = 0;
+  hoverMoveText: string = '';
+  StepTips: string = '操作步骤：将图片拖至禁止角标区域' + '\n' + '预期结果:显示禁止标识,结果为True'
+  uri: string = ""
+  AblockArr: string[] = []
+  BblockArr: string[] = []
+  AVisible: Visibility = Visibility.Visible
   @State dragSuccess: Boolean = false
 
   aboutToAppear(): void {
     FirstDialog.ChooseDialog(this.StepTips, this.name);
     this.getUIContext().getDragController().enableDropDisallowedBadge(true);
   }
-
```

## 50ccc7550ac5
- project: `acts_validator`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-unchanged-state-var` (suggestion)
- file: `entry/src/main/ets/pages/Camera/CameraFlash.ets`:27:3
- patch: shape=`modify`, changed_lines=3

### Before (local context)
```
    21  
    22  @Entry
    23  @Component
    24  struct cameraOrientation {
    25    @State FillColor: string = '#FF000000';
    26    @State name: string = 'CameraFlash';
    27    @State StepTips: string = '测试目的：用于测试相机闪光灯能力\n测试步骤：如果设备存在闪光灯，选择开启，否则选择无闪光灯' + '\n' + '预期结果：操作后闪关灯表现一致';
    28    private tag: string = 'qlw CameraFlash';
    29    @State Vue: boolean = false;
    30    @State isFlash: boolean = false;
    31    private mXComponentController: XComponentController = new XComponentController();
    32    @State captureSession: camera.CaptureSession | undefined = undefined;
    33    @State flashChange: boolean = false;
```

### After (local context)
```
    21  import router from '@ohos.router';
    22  
    23  @Entry
    24  @Component
    25  struct cameraOrientation {
    26    @State FillColor: string = '#FF000000';
    27    @State name: string = 'CameraFlash';
    28    // StepTips 未发生变化，改为普通变量
    29    StepTips: string = '测试目的：用于测试相机闪光灯能力\n测试步骤：如果设备存在闪光灯，选择开启，否则选择无闪光灯' + '\n' + '预期结果：操作后闪关灯表现一致';
    30    private tag: string = 'qlw CameraFlash';
    31    @State Vue: boolean = false;
    32    @State isFlash: boolean = false;
    33    private mXComponentController: XComponentController = new XComponentController();
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/Camera/CameraFlash.ets
+++ after/entry/src/main/ets/pages/Camera/CameraFlash.ets
@@ -1,3 +1,4 @@
+ */
 import camera from '@ohos.multimedia.camera'
 import Logger from '../model/Logger'
 import CameraService from '../model/CameraService'
@@ -10,7 +11,8 @@
 struct cameraOrientation {
   @State FillColor: string = '#FF000000';
   @State name: string = 'CameraFlash';
-  @State StepTips: string = '测试目的：用于测试相机闪光灯能力\n测试步骤：如果设备存在闪光灯，选择开启，否则选择无闪光灯' + '\n' + '预期结果：操作后闪关灯表现一致';
+  // StepTips 未发生变化，改为普通变量
+  StepTips: string = '测试目的：用于测试相机闪光灯能力\n测试步骤：如果设备存在闪光灯，选择开启，否则选择无闪光灯' + '\n' + '预期结果：操作后闪关灯表现一致';
   private tag: string = 'qlw CameraFlash';
   @State Vue: boolean = false;
   @State isFlash: boolean = false;
@@ -21,5 +23,3 @@
 
   async aboutToAppear() {
     await FirstDialog.ChooseDialog(this.StepTips, this.name);
-  }
-
```

## 8d6af38998ed
- project: `acts_validator`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-container-without-property` (suggestion)
- file: `entry/src/main/ets/pages/Experience/SettingsColdStartTest.ets`:134:13
- patch: shape=`modify`, changed_lines=24

### Before (local context)
```
   128              Row() {
   129                Text('\n' + '\n' + `根据以下操作步骤完成测试` + '\n' + '\n' + '待测试应用为：' + this.appName + '\n' + '\n')
   130                  .fontColor(Color.White)
   131                  .fontSize('24fp')
   132              }
   133  
   134              Row() {
   135                Text(`测试步骤:` + '\n' + '\n' + '\n' + '\n' + `1.点击开始键进入系统桌面` + '\n' + '\n'
   136                  + `2.清空除validator外的后台应用` + '\n' + '\n' + `3.点击悬浮球开始测试` + '\n' + '\n' + `4.快速点击桌面设置应用`
   137                  + '\n' + '\n' + '5.待悬浮球倒计时结束显示为done后返回validator应用' + '\n' + '\n' + `6.点击结束观察测试结果`
   138                  + '\n' + '\n' + `7.若冷启动在规定时间内完成则通过测试` + '\n' + '\n' + `注意事项:` + '\n' + '\n' +
   139                  `※1.若悬浮球显示连接失败，需重启设备并在run.bat中输入run validator拉起测试` + '\n' + '\n' + `※2.双击悬浮球中断测试，长按悬浮球提前结束测试` + '\n' + '\n' +
   140                  `※3.若无可测试应用，点击无可测试应用按钮`)
```

### After (local context)
```
   128  
   129  
   130              Text('\n' + '\n' + `根据以下操作步骤完成测试` + '\n' + '\n' + '待测试应用为：' + this.appName + '\n' + '\n')
   131                .fontColor(Color.White)
   132                .fontSize('24fp')
   133              
   134  
   135  
   136              Text(`测试步骤:` + '\n' + '\n' + '\n' + '\n' + `1.点击开始键进入系统桌面` + '\n' + '\n'
   137                + `2.清空除validator外的后台应用` + '\n' + '\n' + `3.点击悬浮球开始测试` + '\n' + '\n' + `4.快速点击桌面设置应用`
   138                + '\n' + '\n' + '5.待悬浮球倒计时结束显示为done后返回validator应用' + '\n' + '\n' + `6.点击结束观察测试结果`
   139                + '\n' + '\n' + `7.若冷启动在规定时间内完成则通过测试` + '\n' + '\n' + `注意事项:` + '\n' + '\n' +
   140                `※1.若悬浮球显示连接失败，需重启设备并在run.bat中输入run validator拉起测试` + '\n' + '\n' + `※2.双击悬浮球中断测试，长按悬浮球提前结束测试` + '\n' + '\n' +
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/Experience/SettingsColdStartTest.ets
+++ after/entry/src/main/ets/pages/Experience/SettingsColdStartTest.ets
@@ -1,25 +1,25 @@
+                      url: 'pages/Experience/Experience_index',
                       params: { result: 'false', title: this.name }
                     })
                   })
               }
             }
 
-            Row() {
-              Text('\n' + '\n' + `根据以下操作步骤完成测试` + '\n' + '\n' + '待测试应用为：' + this.appName + '\n' + '\n')
-                .fontColor(Color.White)
-                .fontSize('24fp')
-            }
+
+            Text('\n' + '\n' + `根据以下操作步骤完成测试` + '\n' + '\n' + '待测试应用为：' + this.appName + '\n' + '\n')
+              .fontColor(Color.White)
+              .fontSize('24fp')
+            
+
+
+            Text(`测试步骤:` + '\n' + '\n' + '\n' + '\n' + `1.点击开始键进入系统桌面` + '\n' + '\n'
+              + `2.清空除validator外的后台应用` + '\n' + '\n' + `3.点击悬浮球开始测试` + '\n' + '\n' + `4.快速点击桌面设置应用`
+              + '\n' + '\n' + '5.待悬浮球倒计时结束显示为done后返回validator应用' + '\n' + '\n' + `6.点击结束观察测试结果`
+              + '\n' + '\n' + `7.若冷启动在规定时间内完成则通过测试` + '\n' + '\n' + `注意事项:` + '\n' + '\n' +
+              `※1.若悬浮球显示连接失败，需重启设备并在run.bat中输入run validator拉起测试` + '\n' + '\n' + `※2.双击悬浮球中断测试，长按悬浮球提前结束测试` + '\n' + '\n' +
+              `※3.若无可测试应用，点击无可测试应用按钮`)
+              .fontColor(Color.White)
+              .fontSize('20fp')
+            
 
             Row() {
-              Text(`测试步骤:` + '\n' + '\n' + '\n' + '\n' + `1.点击开始键进入系统桌面` + '\n' + '\n'
-                + `2.清空除validator外的后台应用` + '\n' + '\n' + `3.点击悬浮球开始测试` + '\n' + '\n' + `4.快速点击桌面设置应用`
-                + '\n' + '\n' + '5.待悬浮球倒计时结束显示为done后返回validator应用' + '\n' + '\n' + `6.点击结束观察测试结果`
-                + '\n' + '\n' + `7.若冷启动在规定时间内完成则通过测试` + '\n' + '\n' + `注意事项:` + '\n' + '\n' +
-                `※1.若悬浮球显示连接失败，需重启设备并在run.bat中输入run validator拉起测试` + '\n' + '\n' + `※2.双击悬浮球中断测试，长按悬浮球提前结束测试` + '\n' + '\n' +
-                `※3.若无可测试应用，点击无可测试应用按钮`)
-                .fontColor(Color.White)
-                .fontSize('20fp')
-            }
-
-            Row() {
-              Column() {
```

## 152c7dcb565e
- project: `acts_validator`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-container-without-property` (suggestion)
- file: `entry/src/main/ets/pages/MultimodalInput/KeyBoard.ets`:38:9
- patch: shape=`modify`, changed_lines=35

### Before (local context)
```
    32    }
    33  
    34    @Builder
    35    specificNoParam() {
    36      Column() {
    37        Scroll() {
    38          Column() {
    39            Column() {
    40              Text('请在下面通过外接/系统软键盘输入：pass')
    41                .fontSize(24)
    42              TextInput({ text: this.text, placeholder: '请输入' })
    43                .placeholderColor(Color.Grey)
    44                .placeholderFont({ size: 14, weight: 400 })
```

### After (local context)
```
    32    async aboutToAppear() {
    33      await FirstDialog.ChooseDialog(this.StepTips, this.name);
    34    }
    35  
    36    @Builder
    37    specificNoParam() {
    38      Column() {
    39        Scroll() {
    40  
    41          Column() {
    42            Text('请在下面通过外接/系统软键盘输入：pass')
    43              .fontSize(24)
    44            TextInput({ text: this.text, placeholder: '请输入' })
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/MultimodalInput/KeyBoard.ets
+++ after/entry/src/main/ets/pages/MultimodalInput/KeyBoard.ets
@@ -1,4 +1,6 @@
-  @State StepTips: string = '操作步骤：请通过外接键盘或者软键盘，在输入框中输入上面的字母' + '\n' + '预期结果：在输入框中能输入pass则通过';
+  @State name: string = 'KeyBoard';
+  // 直接使用一般变量即可
+  StepTips: string = '操作步骤：请通过外接键盘或者软键盘，在输入框中输入上面的字母' + '\n' + '预期结果：在输入框中能输入pass则通过';
   @State Vue: boolean = false;
   @State text: string = '';
 
@@ -10,16 +12,14 @@
   specificNoParam() {
     Column() {
       Scroll() {
+
         Column() {
-          Column() {
-            Text('请在下面通过外接/系统软键盘输入：pass')
-              .fontSize(24)
-            TextInput({ text: this.text, placeholder: '请输入' })
-              .placeholderColor(Color.Grey)
-              .placeholderFont({ size: 14, weight: 400 })
-              .caretColor(Color.Blue)
-              .width(400)
-              .height(40)
-              .margin(20)
-              .fontSize(14)
-              .fontColor(Color.Black)
+          Text('请在下面通过外接/系统软键盘输入：pass')
+            .fontSize(24)
+          TextInput({ text: this.text, placeholder: '请输入' })
+            .placeholderColor(Color.Grey)
+            .placeholderFont({ size: 14, weight: 400 })
+            .caretColor(Color.Blue)
+            .width(400)
+            .height(40)
+            .margin(20)
```

## 9688e10a58bd
- project: `acts_validator`
- round_fixed: `1` (early)
- rule: `performance/hp-performance-no-closures` (suggestion)
- file: `entry/src/main/ets/pages/Notification/Notification_index.ets`:384:3
- patch: shape=`modify`, changed_lines=13

### Before (local context)
```
   378        return;
   379      }
   380    }
   381  }
   382  
   383  function filewrite(name1: string, results: string, titles: string) {
   384    let fd = fs.openSync(txtPath, fs.OpenMode.READ_WRITE | fs.OpenMode.CREATE);
   385    let buf = new ArrayBuffer(4096);
   386    let RD = fs.readSync(fd.fd, buf);
   387    let uint8Array = new Uint8Array(buf, 0, RD);
   388    console.info("RRRRRRRRRRd" + RD);
   389    let report: ESObject = String.fromCharCode(...uint8Array);
   390    let WriteTitle = (titles).toString();
```

### After (local context)
```
   378        }
   379        return;
   380      }
   381    }
   382  }
   383  
   384  function filewrite(name1: string, results: string, titles: string, reportPath: string) {
   385    let fd = fs.openSync(reportPath, fs.OpenMode.READ_WRITE | fs.OpenMode.CREATE);
   386    let buf = new ArrayBuffer(4096);
   387    let RD = fs.readSync(fd.fd, buf);
   388    let uint8Array = new Uint8Array(buf, 0, RD);
   389    console.info("RRRRRRRRRRd" + RD);
   390    let report: ESObject = String.fromCharCode(...uint8Array);
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/Notification/Notification_index.ets
+++ after/entry/src/main/ets/pages/Notification/Notification_index.ets
@@ -1,3 +1,4 @@
+        xmlfd = fs.openSync(xmlPath, fs.OpenMode.READ_WRITE | fs.OpenMode.CREATE);
         fs.writeSync(xmlfd.fd, serializerStr);
       } catch (err) {
         console.error(TAG, "read xmlPath =" + xmlPath + "error:" + err);
@@ -9,8 +10,8 @@
   }
 }
 
-function filewrite(name1: string, results: string, titles: string) {
-  let fd = fs.openSync(txtPath, fs.OpenMode.READ_WRITE | fs.OpenMode.CREATE);
+function filewrite(name1: string, results: string, titles: string, reportPath: string) {
+  let fd = fs.openSync(reportPath, fs.OpenMode.READ_WRITE | fs.OpenMode.CREATE);
   let buf = new ArrayBuffer(4096);
   let RD = fs.readSync(fd.fd, buf);
   let uint8Array = new Uint8Array(buf, 0, RD);
@@ -22,4 +23,3 @@
   let Index: number = report.indexOf(WriteTitle);
   let Log = (titles + ";" + results + ";").toString();
   if (Index == -1) {
-    fs.writeSync(fd.fd, Log);
```

## 559726473fe1
- project: `applications_permission_manager`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-use-local-var-to-replace-state-var` (warn)
- file: `permissionmanager/src/main/ets/pages/application-tertiary.ets`:141:9
- patch: shape=`modify`, changed_lines=148

### Before (local context)
```
   135        }
   136      }
   137      if (this.currentGroup === 'LOCATION') {
   138        try {
   139          let acManager = abilityAccessCtrl.createAtManager();
   140          let fuzzyState = acManager.verifyAccessTokenSync(this.tokenId, Permission.APPROXIMATELY_LOCATION);
   141          fuzzyState === abilityAccessCtrl.GrantStatus.PERMISSION_GRANTED ?
   142            this.selected = Constants.PERMISSION_ALLOWED_ONLY_DURING_USE : null;
   143          let accurateStatus = acManager.verifyAccessTokenSync(this.tokenId, Permission.LOCATION);
   144          this.accurateIsOn = (accurateStatus == abilityAccessCtrl.GrantStatus.PERMISSION_GRANTED) ? true : false;
   145          let backgroundState = acManager.verifyAccessTokenSync(this.tokenId, Permission.LOCATION_IN_BACKGROUND);
   146          backgroundState === abilityAccessCtrl.GrantStatus.PERMISSION_GRANTED ?
   147            this.selected = Constants.PERMISSION_ALLOW : null;
```

### After (local context)
```
   135          Log.error('getPermissionFlags error: ' + JSON.stringify(err));
   136        }
   137      }
   138      if (this.currentGroup === 'LOCATION') {
   139        try {
   140          let acManager = abilityAccessCtrl.createAtManager();
   141          let fuzzyState = acManager.verifyAccessTokenSync(this.tokenId, Permission.APPROXIMATELY_LOCATION);
   142          fuzzyState === abilityAccessCtrl.GrantStatus.PERMISSION_GRANTED ?
   143            selected = Constants.PERMISSION_ALLOWED_ONLY_DURING_USE : null;
   144          let accurateStatus = acManager.verifyAccessTokenSync(this.tokenId, Permission.LOCATION);
   145          this.accurateIsOn = (accurateStatus == abilityAccessCtrl.GrantStatus.PERMISSION_GRANTED) ? true : false;
   146          let backgroundState = acManager.verifyAccessTokenSync(this.tokenId, Permission.LOCATION_IN_BACKGROUND);
   147          backgroundState === abilityAccessCtrl.GrantStatus.PERMISSION_GRANTED ?
```

### Local diff
```diff
--- before/permissionmanager/src/main/ets/pages/application-tertiary.ets
+++ after/permissionmanager/src/main/ets/pages/application-tertiary.ets
@@ -1,3 +1,4 @@
+      try {
         let acManager = abilityAccessCtrl.createAtManager();
         acManager.getPermissionFlags(this.tokenId, Permission.READ_PASTEBOARD).then(flag => {
           flag === Constants.PERMISSION_ALLOW_THIS_TIME ? this.selected = Constants.PERMISSION_ONLY_THIS_TIME : null;
@@ -11,15 +12,14 @@
         let acManager = abilityAccessCtrl.createAtManager();
         let fuzzyState = acManager.verifyAccessTokenSync(this.tokenId, Permission.APPROXIMATELY_LOCATION);
         fuzzyState === abilityAccessCtrl.GrantStatus.PERMISSION_GRANTED ?
-          this.selected = Constants.PERMISSION_ALLOWED_ONLY_DURING_USE : null;
+          selected = Constants.PERMISSION_ALLOWED_ONLY_DURING_USE : null;
         let accurateStatus = acManager.verifyAccessTokenSync(this.tokenId, Permission.LOCATION);
         this.accurateIsOn = (accurateStatus == abilityAccessCtrl.GrantStatus.PERMISSION_GRANTED) ? true : false;
         let backgroundState = acManager.verifyAccessTokenSync(this.tokenId, Permission.LOCATION_IN_BACKGROUND);
         backgroundState === abilityAccessCtrl.GrantStatus.PERMISSION_GRANTED ?
-          this.selected = Constants.PERMISSION_ALLOW : null;
+          selected = Constants.PERMISSION_ALLOW : null;
         acManager.getPermissionFlags(this.tokenId, Permission.APPROXIMATELY_LOCATION ).then(flag => {
           flag === Constants.PERMISSION_ALLOW_THIS_TIME ? this.selected = Constants.PERMISSION_ONLY_THIS_TIME : null;
         })
       } catch (err) {
         Log.error('change location status error: ' + JSON.stringify(err));
-      }
```

## e27585c3dd02
- project: `asn1_ber`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-use-local-var-to-replace-state-var` (warn)
- file: `entry/src/main/ets/pages/Index.ets`:440:5
- patch: shape=`modify`, changed_lines=382

### Before (local context)
```
   434      writer9.writeInt(8388607)
   435      let buffer9: buffer.Buffer = writer9.buffer
   436      this.result += '-->  write 3 byte positive -highest: ' + (buffer9.length == 5) + '\r\n'
   437      this.result += '-->  write 3 byte positive -highest: ' + (buffer9[0] == 0x02) + '\r\n'
   438      this.result += '-->  write 3 byte positive -highest: ' + (buffer9[1] == 0x03) + '\r\n'
   439      this.result += '-->  write 3 byte positive -highest: ' + (buffer9[2] == 0x7f) + '\r\n'
   440      this.result += '-->  write 3 byte positive -highest: ' + (buffer9[3] == 0xff) + '\r\n'
   441      this.result += '-->  write 3 byte positive -highest: ' + (buffer9[4] == 0xff) + '\r\n'
   442  
   443      let writer10 = new BerWriter()
   444      writer10.writeInt(8388608)
   445      let buffer10: buffer.Buffer = writer10.buffer
   446      this.result += '-->  write 4 byte positive -lowest: ' + (buffer10.length == 6) + '\r\n'
```

### After (local context)
```
   434  
   435  
   436      let writer7 = new BerWriter()
   437      writer7.writeInt(32768)
   438      let buffer7: buffer.Buffer = writer7.buffer
   439      tempResult += '-->  write 3 byte positive - lowest: ' + (buffer7.length == 5) + '\r\n'
   440      tempResult += '-->  write 3 byte positive - lowest: ' + (buffer7[0] == 0x02) + '\r\n'
   441      tempResult += '-->  write 3 byte positive - lowest: ' + (buffer7[1] == 0x03) + '\r\n'
   442      tempResult += '-->  write 3 byte positive - lowest: ' + (buffer7[2] == 0x00) + '\r\n'
   443      tempResult += '-->  write 3 byte positive - lowest: ' + (buffer7[3] == 0x80) + '\r\n'
   444      tempResult += '-->  write 3 byte positive - lowest: ' + (buffer7[4] == 0x00) + '\r\n'
   445  
   446      let writer8 = new BerWriter()
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/Index.ets
+++ after/entry/src/main/ets/pages/Index.ets
@@ -1,25 +1,25 @@
-    this.result += '-->  write 3 byte positive - middle: ' + (buffer8[2] == 0x5a) + '\r\n'
-    this.result += '-->  write 3 byte positive - middle: ' + (buffer8[3] == 0x9c) + '\r\n'
-    this.result += '-->  write 3 byte positive - middle: ' + (buffer8[4] == 0x43) + '\r\n'
+    let buffer6: buffer.Buffer = writer6.buffer
+    tempResult += '-->  write 2 byte positive - highest: ' + (buffer6.length == 4) + '\r\n'
+    tempResult += '-->  write 2 byte positive - highest: ' + (buffer6[0] == 0x02) + '\r\n'
+    tempResult += '-->  write 2 byte positive - highest: ' + (buffer6[1] == 0x02) + '\r\n'
+    tempResult += '-->  write 2 byte positive - highest: ' + (buffer6[2] == 0x7f) + '\r\n'
+    tempResult += '-->  write 2 byte positive - highest: ' + (buffer6[3] == 0xff) + '\r\n'
 
 
-    let writer9 = new BerWriter()
-    writer9.writeInt(8388607)
-    let buffer9: buffer.Buffer = writer9.buffer
-    this.result += '-->  write 3 byte positive -highest: ' + (buffer9.length == 5) + '\r\n'
-    this.result += '-->  write 3 byte positive -highest: ' + (buffer9[0] == 0x02) + '\r\n'
-    this.result += '-->  write 3 byte positive -highest: ' + (buffer9[1] == 0x03) + '\r\n'
-    this.result += '-->  write 3 byte positive -highest: ' + (buffer9[2] == 0x7f) + '\r\n'
-    this.result += '-->  write 3 byte positive -highest: ' + (buffer9[3] == 0xff) + '\r\n'
-    this.result += '-->  write 3 byte positive -highest: ' + (buffer9[4] == 0xff) + '\r\n'
+    let writer7 = new BerWriter()
+    writer7.writeInt(32768)
+    let buffer7: buffer.Buffer = writer7.buffer
+    tempResult += '-->  write 3 byte positive - lowest: ' + (buffer7.length == 5) + '\r\n'
+    tempResult += '-->  write 3 byte positive - lowest: ' + (buffer7[0] == 0x02) + '\r\n'
+    tempResult += '-->  write 3 byte positive - lowest: ' + (buffer7[1] == 0x03) + '\r\n'
+    tempResult += '-->  write 3 byte positive - lowest: ' + (buffer7[2] == 0x00) + '\r\n'
+    tempResult += '-->  write 3 byte positive - lowest: ' + (buffer7[3] == 0x80) + '\r\n'
+    tempResult += '-->  write 3 byte positive - lowest: ' + (buffer7[4] == 0x00) + '\r\n'
 
-    let writer10 = new BerWriter()
-    writer10.writeInt(8388608)
-    let buffer10: buffer.Buffer = writer10.buffer
-    this.result += '-->  write 4 byte positive -lowest: ' + (buffer10.length == 6) + '\r\n'
-    this.result += '-->  write 4 byte positive -lowest: ' + (buffer10[0] == 0x02) + '\r\n'
-    this.result += '-->  write 4 byte positive -lowest: ' + (buffer10[1] == 0x04) + '\r\n'
-    this.result += '-->  write 4 byte positive -lowest: ' + (buffer10[2] == 0x00) + '\r\n'
-    this.result += '-->  write 4 byte positive -lowest: ' + (buffer10[3] == 0x80) + '\r\n'
-    this.result += '-->  write 4 byte positive -lowest: ' + (buffer10[4] == 0x00) + '\r\n'
-    this.result += '-->  write 4 byte positive -lowest: ' + (buffer10[5] == 0x00) + '\r\n'
+    let writer8 = new BerWriter()
+    writer8.writeInt(5938243)
+    let buffer8: buffer.Buffer = writer8.buffer
+    tempResult += '-->  write 3 byte positive - middle: ' + (buffer8.length == 5) + '\r\n'
+    tempResult += '-->  write 3 byte positive - middle: ' + (buffer8[0] == 0x02) + '\r\n'
+    tempResult += '-->  write 3 byte positive - middle: ' + (buffer8[1] == 0x03) + '\r\n'
+    tempResult += '-->  write 3 byte positive - middle: ' + (buffer8[2] == 0x5a) + '\r\n'
```

## 0336493f1916
- project: `audio_suite`
- round_fixed: `1` (early)
- rule: `performance/init-list-component` (warn)
- file: `entry/src/main/ets/pages/AudioEdit.ets`:1269:17
- patch: shape=`modify`, changed_lines=6

### Before (local context)
```
  1263                      .margin({ right: $r('app.float.margin_35') })
  1264                  Text(info.songName).margin({ right: 30 }).fontColor(Color.White)
  1265              }
  1266              .justifyContent(FlexAlign.SpaceBetween)
  1267  
  1268              Row() {
  1269                  List() {
  1270                      ForEach(this.songWaveList.get(nodeId)?.nodes ?? [], (item: Node) => {
  1271                          ListItem() {
  1272                              Text(item.type)
  1273                                  .fontSize(14)
  1274                                  .fontColor('#ffffff')
  1275                          }
```

### After (local context)
```
  1263                      .height($r('app.float.height_50'))
  1264                      .margin({ right: $r('app.float.margin_35') })
  1265                  Text(info.songName).margin({ right: 30 }).fontColor(Color.White)
  1266              }
  1267              .justifyContent(FlexAlign.SpaceBetween)
  1268  
  1269              Row() {
  1270                  List().width('100%').height('100%') {
  1271                      ForEach(this.songWaveList.get(nodeId)?.nodes ?? [], (item: Node) => {
  1272                          ListItem() {
  1273                              Text(item.type)
  1274                                  .fontSize(14)
  1275                                  .fontColor('#ffffff')
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/AudioEdit.ets
+++ after/entry/src/main/ets/pages/AudioEdit.ets
@@ -1,3 +1,4 @@
+    @Builder
     songItem(nodeId: string, info: SongInfo) {
         Column() {
             Row() {
@@ -10,7 +11,7 @@
             .justifyContent(FlexAlign.SpaceBetween)
 
             Row() {
-                List() {
+                List().width('100%').height('100%') {
                     ForEach(this.songWaveList.get(nodeId)?.nodes ?? [], (item: Node) => {
                         ListItem() {
                             Text(item.type)
@@ -22,4 +23,3 @@
                         .borderRadius(20)
                         .margin($r('app.float.margin_5'))
                         .padding($r('app.float.margin_5'))
-                        .backgroundColor(item.color)
```

## a86a91016277
- project: `audio_suite`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-container-without-property` (suggestion)
- file: `entry/src/main/ets/pages/customDialog/CheckNodeDialog.ets`:833:21
- patch: shape=`modify`, changed_lines=278

### Before (local context)
```
   827                          Text(`${this.soundFiledType}`)
   828                      }
   829                  }
   830              }
   831              if (this.nodeType === NodeType.ENV) {
   832                  Row() {
   833                      Column() {
   834                          Text(`环境参数：`)
   835                              .fontWeight(FontWeight.Bold)
   836                      }
   837  
   838                      Column() {
   839                          Text(`${this.environmentType}`)
```

### After (local context)
```
   827                            .fontWeight(FontWeight.Bold)
   828                      
   829  
   830  
   831                        Text(`${this.environmentType}`)
   832                      
   833                  }
   834              }
   835              if (this.nodeType === NodeType.VB) {
   836                  Row() {
   837  
   838                        Text(`美化参数：`)
   839                            .fontWeight(FontWeight.Bold)
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/customDialog/CheckNodeDialog.ets
+++ after/entry/src/main/ets/pages/customDialog/CheckNodeDialog.ets
@@ -1,25 +1,25 @@
-                    Column() {
-                        Text(`声场参数：`)
-                            .fontWeight(FontWeight.Bold)
-                    }
-
-                    Column() {
-                        Text(`${this.soundFiledType}`)
-                    }
                 }
             }
             if (this.nodeType === NodeType.ENV) {
                 Row() {
-                    Column() {
-                        Text(`环境参数：`)
-                            .fontWeight(FontWeight.Bold)
-                    }
 
-                    Column() {
-                        Text(`${this.environmentType}`)
-                    }
+                      Text(`环境参数：`)
+                          .fontWeight(FontWeight.Bold)
+                    
+
+
+                      Text(`${this.environmentType}`)
+                    
                 }
             }
             if (this.nodeType === NodeType.VB) {
                 Row() {
-                    Column() {
+
+                      Text(`美化参数：`)
+                          .fontWeight(FontWeight.Bold)
+                    
+
+
+                      Text(`${this.voiceBeautifierType}`)
+                    
+                }
```

## 00859aa70c46
- project: `bluetoothtest`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-no-state-var-access-in-loop` (warn)
- file: `entry/src/main/ets/pages/subAutoTestManager/subGattAutoTestManager/gattClientAutoTestManager.ets`:150:7
- patch: shape=`modify`, changed_lines=4

### Before (local context)
```
   144        if ( typeof GattClientManagerAutoTestCase[ m ].api === "function" ) {
   145          // 使用 TypeScript 的类型断言确保 para 是一个数组
   146          gattClientAutoTestMessage += GattClientManagerAutoTestCase[ m ].api(... interfaces[ m ].para) + "\n";
   147        } else {
   148          console.error(`在 BrAutoTestCase 中找不到方法: ${ GattClientManagerAutoTestCase[ m ] }`);
   149        }
   150        this.changeIndex = m
   151        await sleep(1)
   152        AppStorage.setOrCreate("gattClientAutoTestMessage" , gattClientAutoTestMessage)
   153      }
   154    }
   155  }
   156  
```

### After (local context)
```
   144      for ( let m = 0 ; m < interfaces.length ; m ++ ) {
   145        // 检查 GattClientManagerAutoTestCase 中对应的 api 是否是一个函数
   146        if ( typeof GattClientManagerAutoTestCase[ m ].api === "function" ) {
   147          // 使用 TypeScript 的类型断言确保 para 是一个数组
   148          gattClientAutoTestMessage += GattClientManagerAutoTestCase[ m ].api(... interfaces[ m ].para) + "\n";
   149        } else {
   150          console.error(`在 BrAutoTestCase 中找不到方法: ${ GattClientManagerAutoTestCase[ m ] }`);
   151        }
   152        changeIndex = m
   153        await sleep(1)
   154        AppStorage.setOrCreate("gattClientAutoTestMessage" , gattClientAutoTestMessage)
   155      }
   156      this.changeIndex = changeIndex
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/subAutoTestManager/subGattAutoTestManager/gattClientAutoTestManager.ets
+++ after/entry/src/main/ets/pages/subAutoTestManager/subGattAutoTestManager/gattClientAutoTestManager.ets
@@ -1,7 +1,9 @@
+    let interfaces = GattClientManagerAutoTestCase.map(i => {
       let item = gattClientAutoArray.find(itm => itm.name === i.api.name);
       let para = item ? item.para : []; // 确保 para 总是一个数组
       return { name : i.api.name , para } as BTAutoArrayItem;
     });
+    let changeIndex = this.changeIndex
     for ( let m = 0 ; m < interfaces.length ; m ++ ) {
       // 检查 GattClientManagerAutoTestCase 中对应的 api 是否是一个函数
       if ( typeof GattClientManagerAutoTestCase[ m ].api === "function" ) {
@@ -10,16 +12,14 @@
       } else {
         console.error(`在 BrAutoTestCase 中找不到方法: ${ GattClientManagerAutoTestCase[ m ] }`);
       }
-      this.changeIndex = m
+      changeIndex = m
       await sleep(1)
       AppStorage.setOrCreate("gattClientAutoTestMessage" , gattClientAutoTestMessage)
     }
+    this.changeIndex = changeIndex
   }
 }
 
 async function sleep(time: number): Promise<void> {
   return new Promise<void>((resolve , reject) => {
     setTimeout(() => {
-      resolve();
-    } , time * 1000);
-  });
```

## 1d7fe9d8ecec
- project: `bluetoothtest`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-unchanged-state-var` (suggestion)
- file: `entry/src/main/ets/pages/subBenchmarkTest/gattClientVelocityBenchmarkTest.ets`:45:3
- patch: shape=`modify`, changed_lines=11

### Before (local context)
```
    39    @State gatt_Cycle_index: number = 10
    40    @StorageLink("gattClientBenchmarkTestMessage") gattClientBenchmarkTestMessage: string = ""
    41    private peripheralDeviceId = "08:FB:EA:1B:3C:63"
    42    @State serviceUUID: string = "00001801-0000-1000-8000-00805f9b34fb";
    43    @State characteristicUUID: string = "00002b29-0000-1000-8000-00805f9b34fb";
    44    @State characteristicValue: string = "CccValue";
    45    @State descriptorUUID: string = "00002902-0000-1000-8000-00805f9b34fb";
    46    @State descriptorValue: string = "DesValue";
    47  
    48    aboutToAppear() {
    49      AppStorage.setOrCreate("peripheralDeviceId" , this.peripheralDeviceId);
    50      AppStorage.setOrCreate("serviceUUID" , this.serviceUUID);
    51      AppStorage.setOrCreate("characteristicUUID" , this.characteristicUUID);
```

### After (local context)
```
    39    changeIndex: number = - 1
    40    gatt_Cycle_index: number = 10
    41    @StorageLink("gattClientBenchmarkTestMessage") gattClientBenchmarkTestMessage: string = ""
    42    private peripheralDeviceId = "08:FB:EA:1B:3C:63"
    43    serviceUUID: string = "00001801-0000-1000-8000-00805f9b34fb";
    44    characteristicUUID: string = "00002b29-0000-1000-8000-00805f9b34fb";
    45    characteristicValue: string = "CccValue";
    46    descriptorUUID: string = "00002902-0000-1000-8000-00805f9b34fb";
    47    descriptorValue: string = "DesValue";
    48  
    49    aboutToAppear() {
    50      AppStorage.setOrCreate("peripheralDeviceId" , this.peripheralDeviceId);
    51      AppStorage.setOrCreate("serviceUUID" , this.serviceUUID);
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/subBenchmarkTest/gattClientVelocityBenchmarkTest.ets
+++ after/entry/src/main/ets/pages/subBenchmarkTest/gattClientVelocityBenchmarkTest.ets
@@ -1,17 +1,18 @@
+ */
 
 @Entry
 @Component
 struct GattClientVelocityBenchmarkTest {
   private testItem: TestData = (router.getParams() as myParams).testItem
-  @State changeIndex: number = - 1
-  @State gatt_Cycle_index: number = 10
+  changeIndex: number = - 1
+  gatt_Cycle_index: number = 10
   @StorageLink("gattClientBenchmarkTestMessage") gattClientBenchmarkTestMessage: string = ""
   private peripheralDeviceId = "08:FB:EA:1B:3C:63"
-  @State serviceUUID: string = "00001801-0000-1000-8000-00805f9b34fb";
-  @State characteristicUUID: string = "00002b29-0000-1000-8000-00805f9b34fb";
-  @State characteristicValue: string = "CccValue";
-  @State descriptorUUID: string = "00002902-0000-1000-8000-00805f9b34fb";
-  @State descriptorValue: string = "DesValue";
+  serviceUUID: string = "00001801-0000-1000-8000-00805f9b34fb";
+  characteristicUUID: string = "00002b29-0000-1000-8000-00805f9b34fb";
+  characteristicValue: string = "CccValue";
+  descriptorUUID: string = "00002902-0000-1000-8000-00805f9b34fb";
+  descriptorValue: string = "DesValue";
 
   aboutToAppear() {
     AppStorage.setOrCreate("peripheralDeviceId" , this.peripheralDeviceId);
@@ -22,4 +23,3 @@
     AppStorage.setOrCreate("descriptorValue" , this.descriptorValue);
   }
 
-  build() {
```

## b678452963d3
- project: `bluetoothtest`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-redundant-state-var` (suggestion)
- file: `entry/src/main/ets/pages/subManualApiTest/subBrTest/deviceFound.ets`:42:3
- patch: shape=`modify`, changed_lines=21

### Before (local context)
```
    36  
    37  @Entry
    38  @Component
    39  struct DeviceFound {
    40    private PAGE_TAG = ConfigData.TAG + 'Bluetooth page '
    41    private deviceController: BluetoothDeviceController = new BluetoothDeviceController();
    42    @State message: string = 'BrTest';
    43    @State bluetoothSwitch: boolean = false;
    44    @State scanSwitch: boolean = false;
    45    @StorageLink('bluetoothIsOn') isOn: boolean = false;
    46    @StorageLink('bluetoothToggleEnabled') bluetoothToggleEnabled: boolean = true;
    47    @StorageLink('bluetoothLocalName') localName: string = '';
    48  
```

### After (local context)
```
    36  
    37  @Entry
    38  @Component
    39  struct DeviceFound {
    40    private PAGE_TAG = ConfigData.TAG + 'Bluetooth page '
    41    private deviceController: BluetoothDeviceController = new BluetoothDeviceController();
    42    @StorageLink('bluetoothIsOn') isOn: boolean = false;
    43    @StorageLink('bluetoothToggleEnabled') bluetoothToggleEnabled: boolean = true;
    44    @StorageLink('bluetoothLocalName') localName: string = '';
    45  
    46    aboutToAppear(): void {
    47      LogUtil.log(this.PAGE_TAG + 'aboutToAppear in : isOn = ' + this.isOn)
    48      this.deviceController
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/subManualApiTest/subBrTest/deviceFound.ets
+++ after/entry/src/main/ets/pages/subManualApiTest/subBrTest/deviceFound.ets
@@ -10,9 +10,6 @@
 struct DeviceFound {
   private PAGE_TAG = ConfigData.TAG + 'Bluetooth page '
   private deviceController: BluetoothDeviceController = new BluetoothDeviceController();
-  @State message: string = 'BrTest';
-  @State bluetoothSwitch: boolean = false;
-  @State scanSwitch: boolean = false;
   @StorageLink('bluetoothIsOn') isOn: boolean = false;
   @StorageLink('bluetoothToggleEnabled') bluetoothToggleEnabled: boolean = true;
   @StorageLink('bluetoothLocalName') localName: string = '';
@@ -23,3 +20,6 @@
       .initData()
       .subscribe();
     LogUtil.log(this.PAGE_TAG + 'aboutToAppear out : isOn = ' + this.isOn)
+  }
+
+  onPageShow(): void {
```

## 556a9a658686
- project: `bluetoothtest`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-unchanged-state-var` (suggestion)
- file: `entry/src/main/ets/pages/subManualApiTestManager/bleTestManager.ets`:39:3
- patch: shape=`delete`, changed_lines=20

### Before (local context)
```
    33  struct BleTestManager {
    34    private testItem: TestData = (router.getParams() as myParams).testItem;
    35    @State showList: boolean = true;
    36    @State message: string = 'BleTest Result';
    37    @State peripheralDeviceId: string = '6c:96:d7:3d:87:6f'; // 88:36:CF:09:C1:90
    38    @State currentClick: number = - 1;
    39    @State btOnBLEDeviceFind: string = 'on("BLEDeviceFind"): void';
    40    @State isBLEDeviceFindClick: boolean = false;
    41  
    42    // input ble scan parameters:
    43    /*ScanFilter*/
    44    @State cbxBleScanFilter: boolean = false;
    45    @State txtScanFilter_deviceId: string = '6c:96:d7:3d:87:6f';
```

### After (local context)
```
    33  struct BleTestManager {
    34    private testItem: TestData = (router.getParams() as myParams).testItem;
    35    @State showList: boolean = true;
    36    @State peripheralDeviceId: string = '6c:96:d7:3d:87:6f'; // 88:36:CF:09:C1:90
    37  
    38    // input ble scan parameters:
    39    /*ScanFilter*/
    40    @State txtScanFilter_deviceId: string = '6c:96:d7:3d:87:6f';
    41    @State txtScanFilter_name: string = "dudu-tiger";
    42    @State txtScanFilter_serviceUuid: string = "0000180A-0000-1000-8000-00805f9b34fb";
    43  
    44    /*ScanOptions*/
    45    @StorageLink('bleAvailableDevices') availBleDeviceIds: string[] = [];
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/subManualApiTestManager/bleTestManager.ets
+++ after/entry/src/main/ets/pages/subManualApiTestManager/bleTestManager.ets
@@ -7,19 +7,19 @@
 struct BleTestManager {
   private testItem: TestData = (router.getParams() as myParams).testItem;
   @State showList: boolean = true;
-  @State message: string = 'BleTest Result';
   @State peripheralDeviceId: string = '6c:96:d7:3d:87:6f'; // 88:36:CF:09:C1:90
-  @State currentClick: number = - 1;
-  @State btOnBLEDeviceFind: string = 'on("BLEDeviceFind"): void';
-  @State isBLEDeviceFindClick: boolean = false;
 
   // input ble scan parameters:
   /*ScanFilter*/
-  @State cbxBleScanFilter: boolean = false;
   @State txtScanFilter_deviceId: string = '6c:96:d7:3d:87:6f';
   @State txtScanFilter_name: string = "dudu-tiger";
   @State txtScanFilter_serviceUuid: string = "0000180A-0000-1000-8000-00805f9b34fb";
-  @State curScanFilters: Array<bluetoothManager.ScanFilter> = [];
 
   /*ScanOptions*/
-  @State cbxBleScanOptions: boolean = false;
+  @StorageLink('bleAvailableDevices') availBleDeviceIds: string[] = [];
+  @StorageLink('OnBLEDeviceFind') On_off_BLEDeviceFind: boolean = false;
+
+  aboutToAppear() {
+    AppStorage.setOrCreate("peripheralDeviceId" , this.peripheralDeviceId)
+    AppStorage.setOrCreate('txtScanFilterDeviceId' , this.txtScanFilter_deviceId);
+    AppStorage.setOrCreate('txtScanFilterName' , this.txtScanFilter_name);
```

## a0641fe87d6a
- project: `bluetoothtest`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-unchanged-state-var` (suggestion)
- file: `entry/src/main/ets/pages/subManualApiTestManager/bleTestManager.ets`:47:3
- patch: shape=`delete`, changed_lines=20

### Before (local context)
```
    41  
    42    // input ble scan parameters:
    43    /*ScanFilter*/
    44    @State cbxBleScanFilter: boolean = false;
    45    @State txtScanFilter_deviceId: string = '6c:96:d7:3d:87:6f';
    46    @State txtScanFilter_name: string = "dudu-tiger";
    47    @State txtScanFilter_serviceUuid: string = "0000180A-0000-1000-8000-00805f9b34fb";
    48    @State curScanFilters: Array<bluetoothManager.ScanFilter> = [];
    49  
    50    /*ScanOptions*/
    51    @State cbxBleScanOptions: boolean = false;
    52    @State txtScanOptions_interval: string = "0";
    53    @State txtScanOptions_dutyMode: number = 0; //bluetoothManager.ScanDuty.SCAN_MODE_LOW_POWER;
```

### After (local context)
```
    41    @State txtScanFilter_name: string = "dudu-tiger";
    42    @State txtScanFilter_serviceUuid: string = "0000180A-0000-1000-8000-00805f9b34fb";
    43  
    44    /*ScanOptions*/
    45    @StorageLink('bleAvailableDevices') availBleDeviceIds: string[] = [];
    46    @StorageLink('OnBLEDeviceFind') On_off_BLEDeviceFind: boolean = false;
    47  
    48    aboutToAppear() {
    49      AppStorage.setOrCreate("peripheralDeviceId" , this.peripheralDeviceId)
    50      AppStorage.setOrCreate('txtScanFilterDeviceId' , this.txtScanFilter_deviceId);
    51      AppStorage.setOrCreate('txtScanFilterName' , this.txtScanFilter_name);
    52      AppStorage.setOrCreate('txtScanFilterServiceUuid' , this.txtScanFilter_serviceUuid);
    53      AppStorage.setOrCreate('txtScanOptions_interval' , this.txtScanOptions_interval);
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/subManualApiTestManager/bleTestManager.ets
+++ after/entry/src/main/ets/pages/subManualApiTestManager/bleTestManager.ets
@@ -1,25 +1,25 @@
   @State showList: boolean = true;
-  @State message: string = 'BleTest Result';
   @State peripheralDeviceId: string = '6c:96:d7:3d:87:6f'; // 88:36:CF:09:C1:90
-  @State currentClick: number = - 1;
-  @State btOnBLEDeviceFind: string = 'on("BLEDeviceFind"): void';
-  @State isBLEDeviceFindClick: boolean = false;
 
   // input ble scan parameters:
   /*ScanFilter*/
-  @State cbxBleScanFilter: boolean = false;
   @State txtScanFilter_deviceId: string = '6c:96:d7:3d:87:6f';
   @State txtScanFilter_name: string = "dudu-tiger";
   @State txtScanFilter_serviceUuid: string = "0000180A-0000-1000-8000-00805f9b34fb";
-  @State curScanFilters: Array<bluetoothManager.ScanFilter> = [];
 
   /*ScanOptions*/
-  @State cbxBleScanOptions: boolean = false;
-  @State txtScanOptions_interval: string = "0";
-  @State txtScanOptions_dutyMode: number = 0; //bluetoothManager.ScanDuty.SCAN_MODE_LOW_POWER;
-  @State rd3ScanOptions_dutyModeChecked: boolean = false;
-  @State rd2ScanOptions_dutyModeChecked: boolean = false;
-  @State rd1ScanOptions_dutyModeChecked: boolean = true;
-  @State txtScanOptions_matchMode: number = 1;
-  @State rd1ScanOptions_matchModeChecked: boolean = true;
-  @State rd2ScanOptions_matchModeChecked: boolean = false;
+  @StorageLink('bleAvailableDevices') availBleDeviceIds: string[] = [];
+  @StorageLink('OnBLEDeviceFind') On_off_BLEDeviceFind: boolean = false;
+
+  aboutToAppear() {
+    AppStorage.setOrCreate("peripheralDeviceId" , this.peripheralDeviceId)
+    AppStorage.setOrCreate('txtScanFilterDeviceId' , this.txtScanFilter_deviceId);
+    AppStorage.setOrCreate('txtScanFilterName' , this.txtScanFilter_name);
+    AppStorage.setOrCreate('txtScanFilterServiceUuid' , this.txtScanFilter_serviceUuid);
+    AppStorage.setOrCreate('txtScanOptions_interval' , this.txtScanOptions_interval);
+    AppStorage.setOrCreate('txtScanOptionsDutyMode' , '0');
+    AppStorage.setOrCreate('txtScanOptionsMatchMode' , '0');
+  }
+
+  build() {
+    Column() {
```

## e7b7adcf85e9
- project: `bluetoothtest`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-unchanged-state-var` (suggestion)
- file: `entry/src/main/ets/pages/subManualApiTestManager/bleTestManager.ets`:55:3
- patch: shape=`delete`, changed_lines=20

### Before (local context)
```
    49  
    50    /*ScanOptions*/
    51    @State cbxBleScanOptions: boolean = false;
    52    @State txtScanOptions_interval: string = "0";
    53    @State txtScanOptions_dutyMode: number = 0; //bluetoothManager.ScanDuty.SCAN_MODE_LOW_POWER;
    54    @State rd3ScanOptions_dutyModeChecked: boolean = false;
    55    @State rd2ScanOptions_dutyModeChecked: boolean = false;
    56    @State rd1ScanOptions_dutyModeChecked: boolean = true;
    57    @State txtScanOptions_matchMode: number = 1;
    58    @State rd1ScanOptions_matchModeChecked: boolean = true;
    59    @State rd2ScanOptions_matchModeChecked: boolean = false;
    60    @State curScanOptions: bluetoothManager.ScanOptions = {
    61      interval : 0 ,
```

### After (local context)
```
    49      AppStorage.setOrCreate("peripheralDeviceId" , this.peripheralDeviceId)
    50      AppStorage.setOrCreate('txtScanFilterDeviceId' , this.txtScanFilter_deviceId);
    51      AppStorage.setOrCreate('txtScanFilterName' , this.txtScanFilter_name);
    52      AppStorage.setOrCreate('txtScanFilterServiceUuid' , this.txtScanFilter_serviceUuid);
    53      AppStorage.setOrCreate('txtScanOptions_interval' , this.txtScanOptions_interval);
    54      AppStorage.setOrCreate('txtScanOptionsDutyMode' , '0');
    55      AppStorage.setOrCreate('txtScanOptionsMatchMode' , '0');
    56    }
    57  
    58    build() {
    59      Column() {
    60        Stack({ alignContent : Alignment.TopStart }) {
    61          TestImageDisplay({ testItem : this.testItem })
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/subManualApiTestManager/bleTestManager.ets
+++ after/entry/src/main/ets/pages/subManualApiTestManager/bleTestManager.ets
@@ -1,25 +1,25 @@
-  /*ScanFilter*/
-  @State cbxBleScanFilter: boolean = false;
-  @State txtScanFilter_deviceId: string = '6c:96:d7:3d:87:6f';
-  @State txtScanFilter_name: string = "dudu-tiger";
-  @State txtScanFilter_serviceUuid: string = "0000180A-0000-1000-8000-00805f9b34fb";
-  @State curScanFilters: Array<bluetoothManager.ScanFilter> = [];
 
   /*ScanOptions*/
-  @State cbxBleScanOptions: boolean = false;
-  @State txtScanOptions_interval: string = "0";
-  @State txtScanOptions_dutyMode: number = 0; //bluetoothManager.ScanDuty.SCAN_MODE_LOW_POWER;
-  @State rd3ScanOptions_dutyModeChecked: boolean = false;
-  @State rd2ScanOptions_dutyModeChecked: boolean = false;
-  @State rd1ScanOptions_dutyModeChecked: boolean = true;
-  @State txtScanOptions_matchMode: number = 1;
-  @State rd1ScanOptions_matchModeChecked: boolean = true;
-  @State rd2ScanOptions_matchModeChecked: boolean = false;
-  @State curScanOptions: bluetoothManager.ScanOptions = {
-    interval : 0 ,
-    dutyMode : 0 ,
-    matchMode : 1
-  }
   @StorageLink('bleAvailableDevices') availBleDeviceIds: string[] = [];
   @StorageLink('OnBLEDeviceFind') On_off_BLEDeviceFind: boolean = false;
 
+  aboutToAppear() {
+    AppStorage.setOrCreate("peripheralDeviceId" , this.peripheralDeviceId)
+    AppStorage.setOrCreate('txtScanFilterDeviceId' , this.txtScanFilter_deviceId);
+    AppStorage.setOrCreate('txtScanFilterName' , this.txtScanFilter_name);
+    AppStorage.setOrCreate('txtScanFilterServiceUuid' , this.txtScanFilter_serviceUuid);
+    AppStorage.setOrCreate('txtScanOptions_interval' , this.txtScanOptions_interval);
+    AppStorage.setOrCreate('txtScanOptionsDutyMode' , '0');
+    AppStorage.setOrCreate('txtScanOptionsMatchMode' , '0');
+  }
+
+  build() {
+    Column() {
+      Stack({ alignContent : Alignment.TopStart }) {
+        TestImageDisplay({ testItem : this.testItem })
+        PageTitle({ testItem : this.testItem })
+      }
+
+      Stack().height("0.5vp").backgroundColor("#000000");
+      Column() {
+        Row() {
```

## 36812cac4695
- project: `bluetoothtest`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-redundant-state-var` (suggestion)
- file: `entry/src/main/ets/pages/subManualApiTestManager/bleTestManager.ets`:58:3
- patch: shape=`delete`, changed_lines=20

### Before (local context)
```
    52    @State txtScanOptions_interval: string = "0";
    53    @State txtScanOptions_dutyMode: number = 0; //bluetoothManager.ScanDuty.SCAN_MODE_LOW_POWER;
    54    @State rd3ScanOptions_dutyModeChecked: boolean = false;
    55    @State rd2ScanOptions_dutyModeChecked: boolean = false;
    56    @State rd1ScanOptions_dutyModeChecked: boolean = true;
    57    @State txtScanOptions_matchMode: number = 1;
    58    @State rd1ScanOptions_matchModeChecked: boolean = true;
    59    @State rd2ScanOptions_matchModeChecked: boolean = false;
    60    @State curScanOptions: bluetoothManager.ScanOptions = {
    61      interval : 0 ,
    62      dutyMode : 0 ,
    63      matchMode : 1
    64    }
```

### After (local context)
```
    52      AppStorage.setOrCreate('txtScanFilterServiceUuid' , this.txtScanFilter_serviceUuid);
    53      AppStorage.setOrCreate('txtScanOptions_interval' , this.txtScanOptions_interval);
    54      AppStorage.setOrCreate('txtScanOptionsDutyMode' , '0');
    55      AppStorage.setOrCreate('txtScanOptionsMatchMode' , '0');
    56    }
    57  
    58    build() {
    59      Column() {
    60        Stack({ alignContent : Alignment.TopStart }) {
    61          TestImageDisplay({ testItem : this.testItem })
    62          PageTitle({ testItem : this.testItem })
    63        }
    64  
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/subManualApiTestManager/bleTestManager.ets
+++ after/entry/src/main/ets/pages/subManualApiTestManager/bleTestManager.ets
@@ -1,25 +1,25 @@
-  @State txtScanFilter_name: string = "dudu-tiger";
-  @State txtScanFilter_serviceUuid: string = "0000180A-0000-1000-8000-00805f9b34fb";
-  @State curScanFilters: Array<bluetoothManager.ScanFilter> = [];
-
-  /*ScanOptions*/
-  @State cbxBleScanOptions: boolean = false;
-  @State txtScanOptions_interval: string = "0";
-  @State txtScanOptions_dutyMode: number = 0; //bluetoothManager.ScanDuty.SCAN_MODE_LOW_POWER;
-  @State rd3ScanOptions_dutyModeChecked: boolean = false;
-  @State rd2ScanOptions_dutyModeChecked: boolean = false;
-  @State rd1ScanOptions_dutyModeChecked: boolean = true;
-  @State txtScanOptions_matchMode: number = 1;
-  @State rd1ScanOptions_matchModeChecked: boolean = true;
-  @State rd2ScanOptions_matchModeChecked: boolean = false;
-  @State curScanOptions: bluetoothManager.ScanOptions = {
-    interval : 0 ,
-    dutyMode : 0 ,
-    matchMode : 1
-  }
-  @StorageLink('bleAvailableDevices') availBleDeviceIds: string[] = [];
   @StorageLink('OnBLEDeviceFind') On_off_BLEDeviceFind: boolean = false;
 
   aboutToAppear() {
     AppStorage.setOrCreate("peripheralDeviceId" , this.peripheralDeviceId)
     AppStorage.setOrCreate('txtScanFilterDeviceId' , this.txtScanFilter_deviceId);
+    AppStorage.setOrCreate('txtScanFilterName' , this.txtScanFilter_name);
+    AppStorage.setOrCreate('txtScanFilterServiceUuid' , this.txtScanFilter_serviceUuid);
+    AppStorage.setOrCreate('txtScanOptions_interval' , this.txtScanOptions_interval);
+    AppStorage.setOrCreate('txtScanOptionsDutyMode' , '0');
+    AppStorage.setOrCreate('txtScanOptionsMatchMode' , '0');
+  }
+
+  build() {
+    Column() {
+      Stack({ alignContent : Alignment.TopStart }) {
+        TestImageDisplay({ testItem : this.testItem })
+        PageTitle({ testItem : this.testItem })
+      }
+
+      Stack().height("0.5vp").backgroundColor("#000000");
+      Column() {
+        Row() {
+          Text("外设MAC:")
+            .fontSize("18vp")
+            .height(40)
```

## 9bc52207aaad
- project: `ohos_cordova`
- round_fixed: `2` (early)
- rule: `performance/js-code-cache-by-precompile-check` (suggestion)
- file: `library/src/main/ets/components/CordovaWeb.ets`:288:5
- patch: shape=`add`, changed_lines=1

### Before (local context)
```
   282          ]
   283        }
   284      }
   285    ]
   286  
   287    build() {
   288      Web({ src: '', controller: this.controller, incognitoMode: this.incognitoMode })
   289        .width('100%')
   290        .height('100%')
   291        .id(this.webId)
   292        .imageAccess(true)
   293        .onlineImageAccess(true)
   294        .onControllerAttached(async () => {
```

### After (local context)
```
   282            { headerKey: 'Last-Modified', headerValue: 'Web, 21 Mar 2024 10:38:41 GMT' }
   283          ]
   284        }
   285      }
   286    ]
   287  
   288    build() {
   289      Web({ src: '', controller: this.controller, incognitoMode: this.incognitoMode })
   290        .width('100%')
   291        .height('100%')
   292        .id(this.webId)
   293        .imageAccess(true)
   294        .onlineImageAccess(true)
```

### Local diff
```diff
--- before/library/src/main/ets/components/CordovaWeb.ets
+++ after/library/src/main/ets/components/CordovaWeb.ets
@@ -1,3 +1,4 @@
+    {
       url: 'https://www.example.com/example.js',
       localPath: 'example.js',
       options: {
@@ -22,4 +23,3 @@
           try {
             this.controller.precompileJavaScript(config.url, content, config.options)
               .then((errCode: number) => {
-                console.log('precompile successfully!' );
```

## f8f956e88613
- project: `ohos_dfu_library`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-unchanged-state-var` (suggestion)
- file: `entry/src/main/ets/pages/Index.ets`:131:3
- patch: shape=`modify`, changed_lines=32

### Before (local context)
```
   125    @State selectDevice: Resource = $r('app.string.select_device_label');
   126    @State bootloaderMsg: Resource = $r('app.string.boot_loader_msg_label', '');
   127    @State dfuInitialized: Resource = $r('app.string.dfu_initial_msg_label', '');
   128    @State uploadingMsg: Resource = $r('app.string.upload_msg_label', '');
   129    @State completeMsg: Resource = $r('app.string.complete_msg_label', '');
   130    @State controlMsg: Resource = $r('app.string.start_button');
   131    @State fileName: string = '...';
   132    @State fileSize: string = '...';
   133    @State deviceName: string = '...';
   134    @State deviceMac: string = '...';
   135    @State otaDeviceName: string = '...';
   136    @State otaDeviceMac: string = '...';
   137    // @State filterValue: string = 'MD2-HA'
```

### After (local context)
```
   125    @State selectFile: Resource = $r('app.string.select_file_label');
   126    @State selectDevice: Resource = $r('app.string.select_device_label');
   127    @State bootloaderMsg: Resource = $r('app.string.boot_loader_msg_label', '');
   128    @State dfuInitialized: Resource = $r('app.string.dfu_initial_msg_label', '');
   129    @State uploadingMsg: Resource = $r('app.string.upload_msg_label', '');
   130    @State completeMsg: Resource = $r('app.string.complete_msg_label', '');
   131    @State controlMsg: Resource = $r('app.string.start_button');
   132    fileName: string = '...';
   133    fileSize: string = '...';
   134    deviceName: string = '...';
   135    deviceMac: string = '...';
   136    // @State filterValue: string = 'MD2-HA'
   137    @State filterValue: string = MULAN_OTA
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/Index.ets
+++ after/entry/src/main/ets/pages/Index.ets
@@ -1,3 +1,4 @@
+const MULAN_OTA = 'MD2-HA'
 
 @Entry
 @Component
@@ -10,16 +11,15 @@
   @State uploadingMsg: Resource = $r('app.string.upload_msg_label', '');
   @State completeMsg: Resource = $r('app.string.complete_msg_label', '');
   @State controlMsg: Resource = $r('app.string.start_button');
-  @State fileName: string = '...';
-  @State fileSize: string = '...';
-  @State deviceName: string = '...';
-  @State deviceMac: string = '...';
-  @State otaDeviceName: string = '...';
-  @State otaDeviceMac: string = '...';
+  fileName: string = '...';
+  fileSize: string = '...';
+  deviceName: string = '...';
+  deviceMac: string = '...';
   // @State filterValue: string = 'MD2-HA'
   @State filterValue: string = MULAN_OTA
   @State filterIndex: number = 0;
-  @State otaFilterValue: string = 'OTA'
+  otaFilterValue: string = 'OTA'
   @State canControl: boolean = true;
   @State canFindDevice: boolean = true;
   private gattClient: ble.GattClientDevice | undefined = undefined;
+  private presolve: ((value: string | PromiseLike<string>) => void) | null = null;
```

## af72f4e01879
- project: `ohos_mail_base`
- round_fixed: `1` (early)
- rule: `performance/js-code-cache-by-precompile-check` (suggestion)
- file: `test/src/main/ets/pages/Index.ets`:140:9
- patch: shape=`modify`, changed_lines=56

### Before (local context)
```
   134              Button(item[0])
   135                .onClick(item[1])
   136            })
   137        }
   138        .width(300)
   139        Column() {
   140          Web({ controller: this.controller, src: '' })
   141            .fileAccess(true)
   142            .javaScriptAccess(true)
   143            .layoutWeight(1)
   144            .onlineImageAccess(true)
   145            .domStorageAccess(true)
   146            .zoomAccess(true)
```

### After (local context)
```
   134          Text(`SMTP: ${this.current.smtp.host}:${this.current.smtp.port} ${this.current.smtp.secure ? 'SSL/TLS' : ''}`)
   135          Text(`IMAP: ${this.current.imap.host}:${this.current.imap.port} ${this.current.imap.secure ? 'SSL/TLS' : ''}`)
   136          Text(`POP3: ${this.current.pop3.host}:${this.current.pop3.port} ${this.current.pop3.secure ? 'SSL/TLS' : ''}`)
   137        }
   138        .width(300)
   139        Column() {
   140          ForEach(this.buttons,
   141            (item: [string, (event: ClickEvent) => Promise<void>], index: number) => {
   142              Button(item[0])
   143                .onClick(item[1])
   144            }, (item, index) => item)
   145        }
   146        .width(300)
```

### Local diff
```diff
--- before/test/src/main/ets/pages/Index.ets
+++ after/test/src/main/ets/pages/Index.ets
@@ -1,3 +1,11 @@
+            // getApp().testConnect().then(s => {
+            //   this.message = s ? '连接成功' : '连接失败';
+            // });
+          })
+        Text(this.message)
+
+        Text(`SMTP: ${this.current.smtp.host}:${this.current.smtp.port} ${this.current.smtp.secure ? 'SSL/TLS' : ''}`)
+        Text(`IMAP: ${this.current.imap.host}:${this.current.imap.port} ${this.current.imap.secure ? 'SSL/TLS' : ''}`)
         Text(`POP3: ${this.current.pop3.host}:${this.current.pop3.port} ${this.current.pop3.secure ? 'SSL/TLS' : ''}`)
       }
       .width(300)
@@ -6,20 +14,12 @@
           (item: [string, (event: ClickEvent) => Promise<void>], index: number) => {
             Button(item[0])
               .onClick(item[1])
-          })
+          }, (item, index) => item)
       }
       .width(300)
       Column() {
         Web({ controller: this.controller, src: '' })
-          .fileAccess(true)
-          .javaScriptAccess(true)
-          .layoutWeight(1)
-          .onlineImageAccess(true)
-          .domStorageAccess(true)
-          .zoomAccess(true)
-          .onLoadIntercept((event) => {
-            logger.info('onLoadIntercept, ' + JSON.stringify(event));
-            if (event.data) {
-              const url = event.data.getRequestUrl();
-              logger.info(url);
-            }
+          .onControllerAttached(async () => {
+            for (const config of this.configs) {
+              let content = getContext().resourceManager.getRawFileContentSync(config.localPath);
+              try {
```

## 900a2eb46f5c
- project: `ohos_mail_base`
- round_fixed: `2` (early)
- rule: `performance/reuse-date-instances-check` (warn)
- file: `lib/src/main/ets/format/msg.ts`:366:28
- patch: shape=`add`, changed_lines=1

### Before (local context)
```
   360              const type = v.getUint16();
   361              const tag = v.getUint16();
   362              if ((tag == 0x0039 || tag == 0x0e06) && type == 0x0040) {
   363                v.skip(4);
   364                const ptypTimeAsNumberLong = v.getBigUint64(); // 从1601年1月1日开始的100纳秒数
   365                const millisecondsSinceEpoch = Number(ptypTimeAsNumberLong / 10000n);
   366                const date = new Date(millisecondsSinceEpoch - 11644473600000); // 从1970年1月1日开始毫秒数
   367                this.receivedDate = date;
   368              } else {
   369                v.skip(4 + 8);
   370              }
   371            }
   372          }
```

### After (local context)
```
   360            while (!v.end()) {
   361              const type = v.getUint16();
   362              const tag = v.getUint16();
   363              if ((tag == 0x0039 || tag == 0x0e06) && type == 0x0040) {
   364                v.skip(4);
   365                const ptypTimeAsNumberLong = v.getBigUint64(); // 从1601年1月1日开始的100纳秒数
   366                const millisecondsSinceEpoch = Number(ptypTimeAsNumberLong / 10000n);
   367                const date = new Date(millisecondsSinceEpoch - 11644473600000); // 从1970年1月1日开始毫秒数
   368                this.receivedDate = date;
   369              } else {
   370                v.skip(4 + 8);
   371              }
   372            }
```

### Local diff
```diff
--- before/lib/src/main/ets/format/msg.ts
+++ after/lib/src/main/ets/format/msg.ts
@@ -1,3 +1,4 @@
+  dispatch(entry: DirectoryEntry): void {
     switch (entry.objectType) {
       case DirectoryObjectType.Stream:
         if (entry.name == '__properties_version1.0') {
@@ -22,4 +23,3 @@
           this.setValue(di);
           if (di.name == "RTF_COMPRESSED") {
             logger.info('RTF_COMPRESSED')
-            let src = di.value as (() => Uint8Array);
```

## 91c8d2f4ed25
- project: `shopping`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-unchanged-state-var` (suggestion)
- file: `entry/src/main/ets/pages/Index.ets`:37:5
- patch: shape=`modify`, changed_lines=14

### Before (local context)
```
    31      @State tabsIndex: number = 0
    32      @State tabTittleData: Array<TabTitleModel> = tabTitleData
    33      @State idx: number = 0
    34      controller: TabsController = new TabsController()
    35      @State opacity1: number = 1
    36      @State width1: number = 100
    37      @State ratio: number = 1
    38      @Provide('pathInfos') pathInfos : NavPathStack = new NavPathStack()
    39      //listener = mediaQuery.matchMediaSync('(orientation:landscape)')
    40  
    41      aboutToAppear() {
    42          //this.listener.on('change', this.onPortrait)
    43      }
```

### After (local context)
```
    31      idx: number = 0
    32      controller: TabsController = new TabsController()
    33      opacity1: number = 1
    34      width1: number = 100
    35      ratio: number = 1
    36      @Provide('pathInfos') pathInfos : NavPathStack = new NavPathStack()
    37  
    38      aboutToAppear() {
    39      }
    40      /*
    41        async onPortrait(mediaQueryResult:mediaquery.MediaQueryResult) {
    42          let result = mediaQueryResult.matches
    43          if (result) {
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/Index.ets
+++ after/entry/src/main/ets/pages/Index.ets
@@ -1,21 +1,17 @@
-import { SearchParam } from '../model/routeModel'
-
 @Entry
 @Component
 struct Index {
-    @State device: string = ''
+    device: string = ''
     @State tabsIndex: number = 0
     @State tabTittleData: Array<TabTitleModel> = tabTitleData
-    @State idx: number = 0
+    idx: number = 0
     controller: TabsController = new TabsController()
-    @State opacity1: number = 1
-    @State width1: number = 100
-    @State ratio: number = 1
+    opacity1: number = 1
+    width1: number = 100
+    ratio: number = 1
     @Provide('pathInfos') pathInfos : NavPathStack = new NavPathStack()
-    //listener = mediaQuery.matchMediaSync('(orientation:landscape)')
 
     aboutToAppear() {
-        //this.listener.on('change', this.onPortrait)
     }
     /*
       async onPortrait(mediaQueryResult:mediaquery.MediaQueryResult) {
@@ -23,3 +19,7 @@
         if (result) {
           this.width1 = 45
         } else {
+          this.width1 = 100
+        }
+      }
+    */
```

## 2f70534efdb1
- project: `shopping`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-unchanged-state-var` (suggestion)
- file: `entry/src/main/ets/pages/homePage/User.ets`:23:3
- patch: shape=`modify`, changed_lines=8

### Before (local context)
```
    17  import { RecordModel, OrderModel, DiscountModel, ServiceModel } from '../../model/homeModel'
    18  import { recordData, orderData, discountData, serviceData } from '../../data/homeData'
    19  import { Core2 } from './core2'
    20  
    21  @Component
    22  export struct User {
    23    @State signName: string = '点这里可以添加个性签名'
    24    @State cardName: string = '用户'
    25    @State url: string = '/resources/common/user.png'
    26    @State record: Array<RecordModel> = recordData
    27    @State orderData: Array<OrderModel> = orderData
    28    @State discountData: Array<DiscountModel> = discountData
    29    @State serviceData: Array<ServiceModel> = serviceData
```

### After (local context)
```
    17  import { RecordModel, OrderModel, DiscountModel, ServiceModel } from '../../model/homeModel'
    18  import { recordData, orderData, discountData, serviceData } from '../../data/homeData'
    19  import { Core2 } from './core2'
    20  
    21  @Component
    22  export struct User {
    23    // signName 未发生变化，改为普通变量
    24    signName: string = '点这里可以添加个性签名'
    25    // cardName 未发生变化，改为普通变量
    26    cardName: string = '用户'
    27    // url 未发生变化，改为普通变量
    28    url: string = '/resources/common/user.png'
    29    @State record: Array<RecordModel> = recordData
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/homePage/User.ets
+++ after/entry/src/main/ets/pages/homePage/User.ets
@@ -10,9 +10,12 @@
 
 @Component
 export struct User {
-  @State signName: string = '点这里可以添加个性签名'
-  @State cardName: string = '用户'
-  @State url: string = '/resources/common/user.png'
+  // signName 未发生变化，改为普通变量
+  signName: string = '点这里可以添加个性签名'
+  // cardName 未发生变化，改为普通变量
+  cardName: string = '用户'
+  // url 未发生变化，改为普通变量
+  url: string = '/resources/common/user.png'
   @State record: Array<RecordModel> = recordData
   @State orderData: Array<OrderModel> = orderData
   @State discountData: Array<DiscountModel> = discountData
@@ -20,6 +23,3 @@
   @Prop num: number
   @Prop ratio: number
 
-  build() {
-    Scroll() {
-      Column() {
```

## 1e925b119247
- project: `shopping`
- round_fixed: `1` (early)
- rule: `performance/multiple-associations-state-var-check` (suggestion)
- file: `entry/src/main/ets/pages/homePage/informance.ets`:25:7
- patch: shape=`modify`, changed_lines=2

### Before (local context)
```
    19  
    20  @Component
    21  export struct Information {
    22  @State color1: string = '#ff0000'
    23  @State color2: string = '#00ff00'
    24  @State information: Array<InformationModel> = informationData
    25  @Prop ratio: number
    26  
    27    build() {
    28      Column() {
    29        Button("CallFetchWithCoroutine")
    30            .backgroundColor(this.color1)
    31            .width(200).height(100)
```

### After (local context)
```
    19  import { callFetchWithEACoroutine, callFetchWithCoroutine, callFetchWithTaskPool} from '../../arkcompilerPOC/concurrency'
    20  
    21  @Component
    22  export struct Information {
    23  @State color1: string = '#ff0000'
    24  @State color2: string = '#00ff00'
    25  information: Array<InformationModel> = informationData
    26  @Prop ratio: number
    27  
    28    build() {
    29      Column() {
    30        Button("CallFetchWithCoroutine")
    31            .backgroundColor(this.color1)
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/homePage/informance.ets
+++ after/entry/src/main/ets/pages/homePage/informance.ets
@@ -1,3 +1,4 @@
+ * See the License for the specific language governing permissions and
  * limitations under the License.
  */
 
@@ -9,7 +10,7 @@
 export struct Information {
 @State color1: string = '#ff0000'
 @State color2: string = '#00ff00'
-@State information: Array<InformationModel> = informationData
+information: Array<InformationModel> = informationData
 @Prop ratio: number
 
   build() {
@@ -22,4 +23,3 @@
             callFetchWithCoroutine();
           })
       Button("CallFetchWithTaskPool")
-          .backgroundColor(this.color2)
```

## 8187d2534739
- project: `wifi_testapp`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-unchanged-state-var` (suggestion)
- file: `entry/src/main/ets/Component/filterTable.ets`:37:3
- patch: shape=`modify`, changed_lines=156

### Before (local context)
```
    31    /*ScanFilter*/
    32    @State cbxBleScanFilter: boolean = false;
    33    @State h_ssid: string = "testApp1";
    34    @State h_securityType: number = 3;
    35    @State h_band: number = 2;
    36    @State h_preSharedKey: string = "12345678";
    37    @State h_maxConn: number = 3;
    38    @State h_channel: number = 36;
    39  
    40    getCurrentState(index: number) {
    41      return this.apiItems[ index ].result
    42    }
    43  
```

### After (local context)
```
    31    // input ble scan parameters:
    32    /*ScanFilter*/
    33    cbxBleScanFilter: boolean = false;
    34    @State h_ssid: string = "testApp1";
    35    @State h_securityType: number = 3;
    36    @State h_band: number = 2;
    37    @State h_preSharedKey: string = "12345678";
    38    h_maxConn: number = 3;
    39    @State h_channel: number = 36;
    40  
    41    getCurrentState(index: number) {
    42      return this.apiItems[ index ].result
    43    }
```

### Local diff
```diff
--- before/entry/src/main/ets/Component/filterTable.ets
+++ after/entry/src/main/ets/Component/filterTable.ets
@@ -1,16 +1,17 @@
+
 @Component
 export struct FilterTable {
   private testItem!: TestData
-  @State apiItems: TestApi[] = initHotspotApIData()
-  @State changeIndex: number = - 1
+  apiItems: TestApi[] = initHotspotApIData()
+  changeIndex: number = - 1
   // input ble scan parameters:
   /*ScanFilter*/
-  @State cbxBleScanFilter: boolean = false;
+  cbxBleScanFilter: boolean = false;
   @State h_ssid: string = "testApp1";
   @State h_securityType: number = 3;
   @State h_band: number = 2;
   @State h_preSharedKey: string = "12345678";
-  @State h_maxConn: number = 3;
+  h_maxConn: number = 3;
   @State h_channel: number = 36;
 
   getCurrentState(index: number) {
@@ -22,4 +23,3 @@
     AppStorage.setOrCreate("h_securityType" , this.h_securityType)
     AppStorage.setOrCreate("h_band" , this.h_band)
     AppStorage.setOrCreate("h_preSharedKey" , this.h_preSharedKey)
-    AppStorage.setOrCreate("h_maxConn" , this.h_maxConn)
```

## acf9c4e942b5
- project: `wifi_testapp`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-redundant-nest-container` (suggestion)
- file: `entry/src/main/ets/Component/filterTable.ets`:56:9
- patch: shape=`modify`, changed_lines=156

### Before (local context)
```
    50      AppStorage.setOrCreate("h_channel" , this.h_channel)
    51    }
    52  
    53    build() {
    54      Scroll() {
    55        Column() {
    56          Column() {
    57            Column() {
    58              Row() {
    59                Text("ssid").fontSize("17vp").width(60);
    60                TextInput({ text : this.h_ssid , placeholder : "testApp1" })
    61                  .fontSize("15vp")
    62                  .onChange((strInput: string) => {
```

### After (local context)
```
    50      AppStorage.setOrCreate("h_maxConn" , this.h_maxConn)
    51      AppStorage.setOrCreate("h_channel" , this.h_channel)
    52    }
    53  
    54    build() {
    55      Scroll() {
    56        Column() {
    57  
    58          Column() {
    59            Row() {
    60              Text("ssid").fontSize("17vp").width(60);
    61              TextInput({ text : this.h_ssid , placeholder : "testApp1" })
    62                .fontSize("15vp")
```

### Local diff
```diff
--- before/entry/src/main/ets/Component/filterTable.ets
+++ after/entry/src/main/ets/Component/filterTable.ets
@@ -1,3 +1,4 @@
+
   aboutToAppear() {
     AppStorage.setOrCreate("h_ssid" , this.h_ssid)
     AppStorage.setOrCreate("h_securityType" , this.h_securityType)
@@ -10,16 +11,15 @@
   build() {
     Scroll() {
       Column() {
+
         Column() {
-          Column() {
-            Row() {
-              Text("ssid").fontSize("17vp").width(60);
-              TextInput({ text : this.h_ssid , placeholder : "testApp1" })
-                .fontSize("15vp")
-                .onChange((strInput: string) => {
-                  this.h_ssid = strInput;
-                  AppStorage.setOrCreate("h_ssid" , this.h_ssid);
-                })
-                .width(ConfigData.WH_80_100)
-                .borderRadius(1)
-            }
+          Row() {
+            Text("ssid").fontSize("17vp").width(60);
+            TextInput({ text : this.h_ssid , placeholder : "testApp1" })
+              .fontSize("15vp")
+              .onChange((strInput: string) => {
+                this.h_ssid = strInput;
+                AppStorage.setOrCreate("h_ssid" , this.h_ssid);
+              })
+              .width(ConfigData.WH_80_100)
+              .borderRadius(1)
```

## 020d2c459fd1
- project: `wifi_testapp`
- round_fixed: `1` (early)
- rule: `performance/monitor-invisible-area-in-image-animation` (warn)
- file: `entry/src/main/ets/Component/imageAnimatorComponent.ets`:29:7
- patch: shape=`modify`, changed_lines=60

### Before (local context)
```
    23    private imageWidth: number | Resource = 0
    24    private imageHeight: number | Resource = 0
    25    private whtl: number | string = ComponentConfig.value_20;
    26  
    27    build() {
    28      Column() {
    29        ImageAnimator()
    30          .images([ {
    31            src : '../entryability/res/image/hdpi/ic_loading01.png' ,
    32            duration : ComponentConfig.DURATION_TIME ,
    33            width : this.whtl ,
    34            height : this.whtl ,
    35            top : this.whtl ,
```

### After (local context)
```
    23    imageHeight: number | Resource = 0,
    24    whtl: number | string = ComponentConfig.value_20) {
    25    Column() {
    26      ImageAnimator()
    27        .images([ {
    28          src : '../entryability/res/image/hdpi/ic_loading01.png' ,
    29          duration : ComponentConfig.DURATION_TIME ,
    30          width : whtl ,
    31          height : whtl ,
    32          top : whtl ,
    33          left : whtl
    34        }, {
    35          src : '../entryability/res/image/hdpi/ic_loading02.png' ,
```

### Local diff
```diff
--- before/entry/src/main/ets/Component/imageAnimatorComponent.ets
+++ after/entry/src/main/ets/Component/imageAnimatorComponent.ets
@@ -2,24 +2,24 @@
  * ImageAnimator component Of WiFi test
  */
 
-@Component
-export default struct ImageAnimatorComponent {
-  private imageWidth: number | Resource = 0
-  private imageHeight: number | Resource = 0
-  private whtl: number | string = ComponentConfig.value_20;
-
-  build() {
-    Column() {
-      ImageAnimator()
-        .images([ {
-          src : '../entryability/res/image/hdpi/ic_loading01.png' ,
-          duration : ComponentConfig.DURATION_TIME ,
-          width : this.whtl ,
-          height : this.whtl ,
-          top : this.whtl ,
-          left : this.whtl
-        }, {
-          src : '../entryability/res/image/hdpi/ic_loading02.png' ,
-          duration : ComponentConfig.DURATION_TIME ,
-          width : this.whtl ,
-          height : this.whtl ,
+@Builder
+export default function ImageAnimatorComponent(imageWidth: number | Resource = 0,
+  imageHeight: number | Resource = 0,
+  whtl: number | string = ComponentConfig.value_20) {
+  Column() {
+    ImageAnimator()
+      .images([ {
+        src : '../entryability/res/image/hdpi/ic_loading01.png' ,
+        duration : ComponentConfig.DURATION_TIME ,
+        width : whtl ,
+        height : whtl ,
+        top : whtl ,
+        left : whtl
+      }, {
+        src : '../entryability/res/image/hdpi/ic_loading02.png' ,
+        duration : ComponentConfig.DURATION_TIME ,
+        width : whtl ,
+        height : whtl ,
+        top : whtl ,
+        left : whtl
+      }, {
```

## b6cd2f9a9292
- project: `wifi_testapp`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-use-row-column-to-replace-flex` (suggestion)
- file: `entry/src/main/ets/Component/scenarioContentTable.ets`:34:7
- patch: shape=`modify`, changed_lines=4

### Before (local context)
```
    28    private scenarioItemsX!: TestScenario[]
    29    @Prop changeIndex: number;
    30    @State localName: string = 'DaYuBlue'
    31  
    32    @Builder IngredientItem(scenarioItem: TestScenario , index: number) {
    33      Stack() {
    34        Flex() {
    35          Flex({ direction : FlexDirection.Column , alignItems : ItemAlign.Start }) {
    36            Row() {
    37              Text(scenarioItem.detail)
    38                .fontSize("17vp")
    39                .margin({ top : "3vp" , bottom : "3vp" , left : "10vp" })
    40                .textAlign(TextAlign.Start)
```

### After (local context)
```
    28    @State scenarioItems: TestScenario[] = [];
    29    private scenarioItemsX!: TestScenario[]
    30    @Prop changeIndex: number;
    31  
    32    @Builder IngredientItem(scenarioItem: TestScenario , index: number) {
    33      Stack() {
    34        Row() {
    35          Flex({ direction : FlexDirection.Column , alignItems : ItemAlign.Start }) {
    36            Row() {
    37              Text(scenarioItem.detail)
    38                .fontSize("17vp")
    39                .margin({ top : "3vp" , bottom : "3vp" , left : "10vp" })
    40                .textAlign(TextAlign.Start)
```

### Local diff
```diff
--- before/entry/src/main/ets/Component/scenarioContentTable.ets
+++ after/entry/src/main/ets/Component/scenarioContentTable.ets
@@ -1,3 +1,4 @@
+/**
  *  Scenario Test ContentTable Component Page Of Wifi test
  */
 
@@ -6,11 +7,10 @@
   @State scenarioItems: TestScenario[] = [];
   private scenarioItemsX!: TestScenario[]
   @Prop changeIndex: number;
-  @State localName: string = 'DaYuBlue'
 
   @Builder IngredientItem(scenarioItem: TestScenario , index: number) {
     Stack() {
-      Flex() {
+      Row() {
         Flex({ direction : FlexDirection.Column , alignItems : ItemAlign.Start }) {
           Row() {
             Text(scenarioItem.detail)
```

## bd66bde14bdd
- project: `wifi_testapp`
- round_fixed: `1` (early)
- rule: `performance/js-code-cache-by-precompile-check` (suggestion)
- file: `entry/src/main/ets/Component/webTitleBar.ets`:266:11
- patch: shape=`modify`, changed_lines=40

### Before (local context)
```
   260    }
   261  
   262    build() {
   263      Tabs({ barPosition : BarPosition.Start , controller : this.browser.tabsController }) {
   264        ForEach(this.browser.webArray , (item: WebKey) => {
   265          TabContent() {
   266            Web({
   267              src : this.isPhone ? $rawfile('phone.html') : $rawfile('pad.html') ,
   268              controller : this.browser.webControllerArray[ item.key ] !== undefined ?
   269              this.browser.webControllerArray[ item.key ].controller : undefined
   270            })
   271              .javaScriptAccess(true)
   272              .fileAccess(true)
```

### After (local context)
```
   260    Time: number = 11
   261    @State fileData: string = "";
   262    onPageBeginNumber: number = 0
   263    onPageEndNumber: number = 0
   264    onProgressChangeNumber: number = 0
   265    @StorageLink("openWebNumbers") openWebNumbers: number = 0
   266    @StorageLink("fsFile") file: fs.File | null = null
   267    isRegistered: boolean = false
   268    testObj: TestObject = {
   269      test : (addr: string) => {
   270        console.log(TAG , `addr= ${ this.browser.tabArrayIndex }`)
   271        this.browser.webControllerArray[ this.browser.tabArrayIndex ].controller.loadUrl({ url : `https://${ addr }` })
   272      } ,
```

### Local diff
```diff
--- before/entry/src/main/ets/Component/webTitleBar.ets
+++ after/entry/src/main/ets/Component/webTitleBar.ets
@@ -1,25 +1,25 @@
+  @State progressStartTime: number = 0;
+  @State progressEndTime: number = 0;
+  @State progressCostTime: number = 0;
+  @State pageStartTime: number = 0;
+  @State pageEndTime: number = 0;
+  @State pageCostTime: number = 0;
+  Time: number = 11
+  @State fileData: string = "";
+  onPageBeginNumber: number = 0
+  onPageEndNumber: number = 0
+  onProgressChangeNumber: number = 0
+  @StorageLink("openWebNumbers") openWebNumbers: number = 0
+  @StorageLink("fsFile") file: fs.File | null = null
+  isRegistered: boolean = false
+  testObj: TestObject = {
+    test : (addr: string) => {
+      console.log(TAG , `addr= ${ this.browser.tabArrayIndex }`)
+      this.browser.webControllerArray[ this.browser.tabArrayIndex ].controller.loadUrl({ url : `https://${ addr }` })
+    } ,
+    searchWord : (word: string) => {
       console.log(`search word= ${ word }`)
       let code = encodeURI(word)
       this.browser.webControllerArray[ this.browser.tabArrayIndex ].controller.loadUrl({
         url : `https://www.bing.com/search?q=${ code }`
       })
-    }
-  }
-
-  build() {
-    Tabs({ barPosition : BarPosition.Start , controller : this.browser.tabsController }) {
-      ForEach(this.browser.webArray , (item: WebKey) => {
-        TabContent() {
-          Web({
-            src : this.isPhone ? $rawfile('phone.html') : $rawfile('pad.html') ,
-            controller : this.browser.webControllerArray[ item.key ] !== undefined ?
-            this.browser.webControllerArray[ item.key ].controller : undefined
-          })
-            .javaScriptAccess(true)
-            .fileAccess(true)
-            .domStorageAccess(true)
-            .userAgent(this.isPhone ? PHONE_USER_AGENT : PAD_USER_AGENT)
-            .onPageBegin((event: EventOnPage) => {
-              console.log(TAG , `onPageBegin= ${ JSON.stringify(event) }`)
-              this.pageStartTime = new Date().getTime()
-              console.log(TAG , "onPageBegin,开始加载页面,开始时间：" + this.pageStartTime)
```

## 5335c3c3013a
- project: `wifi_testapp`
- round_fixed: `1` (early)
- rule: `performance/multiple-associations-state-var-check` (suggestion)
- file: `entry/src/main/ets/entryability/candidateWifiModel/view/ProgressEditPanel.ets`:21:9
- patch: shape=`modify`, changed_lines=58

### Before (local context)
```
    15  
    16  import { CommonConstants } from '../common/constant/CommonConstant';
    17  
    18  @Component
    19  export default struct ProgressEditPanel {
    20    @Link sliderMode: number;
    21    @Prop slidingProgress: number;
    22    onCancel!: () => void;
    23    onClickOK!: (progress: number) => void;
    24  
    25    build() {
    26      Column() {
    27        Row() {
```

### After (local context)
```
    15   */
    16  
    17  import { CommonConstants } from '../common/constant/CommonConstant';
    18  
    19  @Builder
    20  function CustomButtonBuilder(buttonText: Resource, onClick: () => void) {
    21    let buttonColor: Resource = $r('app.color.start_window_background');
    22    Text(buttonText)
    23      .dialogButtonStyle()
    24      .backgroundColor(buttonColor)
    25      .borderRadius(CommonConstants.LIST_RADIUS)
    26      .textAlign(TextAlign.Center)
    27      .onTouch((event: TouchEvent) => {
```

### Local diff
```diff
--- before/entry/src/main/ets/entryability/candidateWifiModel/view/ProgressEditPanel.ets
+++ after/entry/src/main/ets/entryability/candidateWifiModel/view/ProgressEditPanel.ets
@@ -1,3 +1,4 @@
+ *
  * Unless required by applicable law or agreed to in writing, software
  * distributed under the License is distributed on an "AS IS" BASIS,
  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
@@ -7,19 +8,18 @@
 
 import { CommonConstants } from '../common/constant/CommonConstant';
 
-@Component
-export default struct ProgressEditPanel {
-  @Link sliderMode: number;
-  @Prop slidingProgress: number;
-  onCancel!: () => void;
-  onClickOK!: (progress: number) => void;
-
-  build() {
-    Column() {
-      Row() {
-        Slider({
-          value : this.slidingProgress ,
-          min : CommonConstants.SLIDER_MIN_VALUE ,
-          max : CommonConstants.SLIDER_MAX_VALUE ,
-          style : SliderStyle.InSet ,
-          step : CommonConstants.SLIDER_STEP
+@Builder
+function CustomButtonBuilder(buttonText: Resource, onClick: () => void) {
+  let buttonColor: Resource = $r('app.color.start_window_background');
+  Text(buttonText)
+    .dialogButtonStyle()
+    .backgroundColor(buttonColor)
+    .borderRadius(CommonConstants.LIST_RADIUS)
+    .textAlign(TextAlign.Center)
+    .onTouch((event: TouchEvent) => {
+      if (event.type === TouchType.Down) {
+        buttonColor = $r('app.color.custom_button_color');
+      } else if (event.type === TouchType.Up) {
+        buttonColor = $r('app.color.start_window_background');
+      }
+    })
```

## ce82d14acdcd
- project: `wifi_testapp`
- round_fixed: `1` (early)
- rule: `performance/multiple-associations-state-var-check` (suggestion)
- file: `entry/src/main/ets/entryability/candidateWifiModel/view/TargetListItem.ets`:28:9
- patch: shape=`modify`, changed_lines=2

### Before (local context)
```
    22  @Component
    23  export default struct TargetListItem {
    24    private taskItem!: TaskItemBean;
    25    @State latestProgress: number = 0;
    26    @State updateDate: string = '';
    27    @Link selectArr: Array<boolean>;
    28    @Prop isEditMode: boolean;
    29    @Link @Watch('onClickIndexChanged') clickIndex: number;
    30    @State isExpanded: boolean = false;
    31    @Consume overAllProgressChanged: boolean;
    32    @State sliderMode: number = CommonConstants.DEFAULT_SLIDER_MODE;
    33    private index!: number;
    34  
```

### After (local context)
```
    22  
    23  @Component
    24  export default struct TargetListItem {
    25    private taskItem!: TaskItemBean;
    26    @State latestProgress: number = 0;
    27    @State updateDate: string = '';
    28    @Link selectArr: Array<boolean>;
    29    @Prop @Watch('onClickIndexChanged') isEditMode: boolean;
    30    @Link @Watch('onClickIndexChanged') clickIndex: number;
    31    @State isExpanded: boolean = false;
    32    @Consume overAllProgressChanged: boolean;
    33    @State sliderMode: number = CommonConstants.DEFAULT_SLIDER_MODE;
    34    private index!: number;
```

### Local diff
```diff
--- before/entry/src/main/ets/entryability/candidateWifiModel/view/TargetListItem.ets
+++ after/entry/src/main/ets/entryability/candidateWifiModel/view/TargetListItem.ets
@@ -1,3 +1,4 @@
+
 import TaskItemBean from '../common/bean/TaskItemBean';
 import { CommonConstants } from '../common/constant/CommonConstant';
 import ProgressEditPanel from './ProgressEditPanel';
@@ -10,7 +11,7 @@
   @State latestProgress: number = 0;
   @State updateDate: string = '';
   @Link selectArr: Array<boolean>;
-  @Prop isEditMode: boolean;
+  @Prop @Watch('onClickIndexChanged') isEditMode: boolean;
   @Link @Watch('onClickIndexChanged') clickIndex: number;
   @State isExpanded: boolean = false;
   @Consume overAllProgressChanged: boolean;
@@ -22,4 +23,3 @@
     this.updateDate = this.taskItem.updateDate;
   }
 
-  /**
```

## c9c89ccd5d87
- project: `wifi_testapp`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-redundant-state-var` (suggestion)
- file: `entry/src/main/ets/pages/subBenchmarkTest/hotspotBenchmarkTest.ets`:81:3
- patch: shape=`modify`, changed_lines=20

### Before (local context)
```
    75    @State message: string = 'hotspotBenchmarkTest';
    76    private testItem: TestData = (router.getParams() as myParams).testItem
    77    @State changeIndex: number = - 1
    78    @StorageLink("hotspotBenchmarkTestMessage") hotspotBenchmarkTestMessage: string = ""
    79    @State receivedSize: number = 0
    80    @State totalSize: number = 0
    81    @State files: Array<string> = []
    82    @State uploads: Array<string> = []
    83    @StorageLink('hotspotBenchmarkTime') hotspotBenchmarkTime: number = 0;
    84    @State h_ssid: string = "testApp1";
    85    @State h_securityType: number = 3;
    86    @State h_band: number = 2;
    87    @State h_preSharedKey: string = "12345678";
```

### After (local context)
```
    75    @State showList: boolean = false;
    76    message: string = 'hotspotBenchmarkTest';
    77    private testItem: TestData = (router.getParams() as myParams).testItem
    78    changeIndex: number = - 1
    79    @StorageLink("hotspotBenchmarkTestMessage") hotspotBenchmarkTestMessage: string = ""
    80    receivedSize: number = 0
    81    totalSize: number = 0
    82    files: Array<string> = []
    83    uploads: Array<string> = []
    84    @StorageLink('hotspotBenchmarkTime') hotspotBenchmarkTime: number = 0;
    85    h_ssid: string = "testApp1";
    86    h_securityType: number = 3;
    87    h_band: number = 2;
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/subBenchmarkTest/hotspotBenchmarkTest.ets
+++ after/entry/src/main/ets/pages/subBenchmarkTest/hotspotBenchmarkTest.ets
@@ -1,25 +1,25 @@
+ *  classic Hotspot benchmark Test Page of wifi test
  */
 
 @Entry
 @Component
 struct HotspotBenchmarkTest {
   @State showList: boolean = false;
-  @State message: string = 'hotspotBenchmarkTest';
+  message: string = 'hotspotBenchmarkTest';
   private testItem: TestData = (router.getParams() as myParams).testItem
-  @State changeIndex: number = - 1
+  changeIndex: number = - 1
   @StorageLink("hotspotBenchmarkTestMessage") hotspotBenchmarkTestMessage: string = ""
-  @State receivedSize: number = 0
-  @State totalSize: number = 0
-  @State files: Array<string> = []
-  @State uploads: Array<string> = []
+  receivedSize: number = 0
+  totalSize: number = 0
+  files: Array<string> = []
+  uploads: Array<string> = []
   @StorageLink('hotspotBenchmarkTime') hotspotBenchmarkTime: number = 0;
-  @State h_ssid: string = "testApp1";
-  @State h_securityType: number = 3;
-  @State h_band: number = 2;
-  @State h_preSharedKey: string = "12345678";
-  @State h_maxConn: number = 3;
-  @State h_channel: number = 36;
+  h_ssid: string = "testApp1";
+  h_securityType: number = 3;
+  h_band: number = 2;
+  h_preSharedKey: string = "12345678";
+  h_maxConn: number = 3;
+  h_channel: number = 36;
 
   aboutToAppear() {
     AppStorage.setOrCreate("h_ssid" , this.h_ssid)
-    AppStorage.setOrCreate("h_securityType" , this.h_securityType)
```

## 521ce8374cae
- project: `wifi_testapp`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-redundant-state-var` (suggestion)
- file: `entry/src/main/ets/pages/subBenchmarkTest/hotspotManagerBenchmarkTest.ets`:88:3
- patch: shape=`modify`, changed_lines=20

### Before (local context)
```
    82    @State uploads: Array<string> = []
    83    @StorageLink('hotspotManagerBenchmarkTime') hotspotManagerBenchmarkTime: number = 0;
    84    @State h_ssid1: string = "testApp1";
    85    @State h_securityType1: number = 3;
    86    @State h_band1: number = 2;
    87    @State h_preSharedKey1: string = "12345678";
    88    @State h_maxConn1: number = 3;
    89    @State h_channel1: number = 36;
    90  
    91    aboutToAppear() {
    92      AppStorage.setOrCreate("h_ssid1" , this.h_ssid1)
    93      AppStorage.setOrCreate("h_securityType1" , this.h_securityType1)
    94      AppStorage.setOrCreate("h_band1" , this.h_band1)
```

### After (local context)
```
    82    files: Array<string> = []
    83    uploads: Array<string> = []
    84    @StorageLink('hotspotManagerBenchmarkTime') hotspotManagerBenchmarkTime: number = 0;
    85    h_ssid1: string = "testApp1";
    86    h_securityType1: number = 3;
    87    h_band1: number = 2;
    88    h_preSharedKey1: string = "12345678";
    89    h_maxConn1: number = 3;
    90    h_channel1: number = 36;
    91  
    92    aboutToAppear() {
    93      AppStorage.setOrCreate("h_ssid1" , this.h_ssid1)
    94      AppStorage.setOrCreate("h_securityType1" , this.h_securityType1)
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/subBenchmarkTest/hotspotManagerBenchmarkTest.ets
+++ after/entry/src/main/ets/pages/subBenchmarkTest/hotspotManagerBenchmarkTest.ets
@@ -1,17 +1,18 @@
+  message: string = 'hotspotManagerBenchmarkTest';
   private testItem: TestData = (router.getParams() as myParams).testItem
-  @State changeIndex: number = - 1
+  changeIndex: number = - 1
   @StorageLink("hotspotManagerBenchmarkTestMessage") hotspotManagerBenchmarkTestMessage: string = ""
-  @State receivedSize: number = 0
-  @State totalSize: number = 0
-  @State files: Array<string> = []
-  @State uploads: Array<string> = []
+  receivedSize: number = 0
+  totalSize: number = 0
+  files: Array<string> = []
+  uploads: Array<string> = []
   @StorageLink('hotspotManagerBenchmarkTime') hotspotManagerBenchmarkTime: number = 0;
-  @State h_ssid1: string = "testApp1";
-  @State h_securityType1: number = 3;
-  @State h_band1: number = 2;
-  @State h_preSharedKey1: string = "12345678";
-  @State h_maxConn1: number = 3;
-  @State h_channel1: number = 36;
+  h_ssid1: string = "testApp1";
+  h_securityType1: number = 3;
+  h_band1: number = 2;
+  h_preSharedKey1: string = "12345678";
+  h_maxConn1: number = 3;
+  h_channel1: number = 36;
 
   aboutToAppear() {
     AppStorage.setOrCreate("h_ssid1" , this.h_ssid1)
@@ -22,4 +23,3 @@
     AppStorage.setOrCreate("h_channel1" , this.h_channel1)
   }
 
-  build() {
```

## 3bd33124ff09
- project: `wifi_testapp`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-redundant-state-var` (suggestion)
- file: `entry/src/main/ets/pages/subBenchmarkTest/p2pManagerBenchmarkTest.ets`:87:3
- patch: shape=`modify`, changed_lines=19

### Before (local context)
```
    81    @State files: Array<string> = []
    82    @State uploads: Array<string> = []
    83    @StorageLink('p2pManagerBenchmarkTime') p2pManagerBenchmarkTime: number = 0;
    84    @State deviceAddressManager: string = '6c:96:d7:3d:87:6f';
    85    @State netIdManager: number = - 2;
    86    @State passphraseManager: string = "12345678";
    87    @State groupNameManager: string = "testGroup"
    88    @State goBandManager: number = 0;
    89    @State devNameManager: string = "MyTestDevice"
    90  
    91    aboutToAppear() {
    92      AppStorage.setOrCreate("deviceAddressManager" , this.deviceAddressManager)
    93      AppStorage.setOrCreate("netIdManager" , this.netIdManager)
```

### After (local context)
```
    81    files: Array<string> = []
    82    uploads: Array<string> = []
    83    @StorageLink('p2pManagerBenchmarkTime') p2pManagerBenchmarkTime: number = 0;
    84    deviceAddressManager: string = '6c:96:d7:3d:87:6f';
    85    netIdManager: number = - 2;
    86    passphraseManager: string = "12345678";
    87    groupNameManager: string = "testGroup"
    88    goBandManager: number = 0;
    89    devNameManager: string = "MyTestDevice"
    90  
    91    aboutToAppear() {
    92      AppStorage.setOrCreate("deviceAddressManager" , this.deviceAddressManager)
    93      AppStorage.setOrCreate("netIdManager" , this.netIdManager)
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/subBenchmarkTest/p2pManagerBenchmarkTest.ets
+++ after/entry/src/main/ets/pages/subBenchmarkTest/p2pManagerBenchmarkTest.ets
@@ -1,18 +1,18 @@
-  @State message: string = 'P2pManagerBenchmarkTest';
+  message: string = 'P2pManagerBenchmarkTest';
   private testItem: TestData = (router.getParams() as myParams).testItem
-  @State changeIndex: number = - 1
+  changeIndex: number = - 1
   @StorageLink("p2pManagerBenchmarkTestMessage") p2pManagerBenchmarkTestMessage: string = ""
-  @State receivedSize: number = 0
-  @State totalSize: number = 0
-  @State files: Array<string> = []
-  @State uploads: Array<string> = []
+  receivedSize: number = 0
+  totalSize: number = 0
+  files: Array<string> = []
+  uploads: Array<string> = []
   @StorageLink('p2pManagerBenchmarkTime') p2pManagerBenchmarkTime: number = 0;
-  @State deviceAddressManager: string = '6c:96:d7:3d:87:6f';
-  @State netIdManager: number = - 2;
-  @State passphraseManager: string = "12345678";
-  @State groupNameManager: string = "testGroup"
-  @State goBandManager: number = 0;
-  @State devNameManager: string = "MyTestDevice"
+  deviceAddressManager: string = '6c:96:d7:3d:87:6f';
+  netIdManager: number = - 2;
+  passphraseManager: string = "12345678";
+  groupNameManager: string = "testGroup"
+  goBandManager: number = 0;
+  devNameManager: string = "MyTestDevice"
 
   aboutToAppear() {
     AppStorage.setOrCreate("deviceAddressManager" , this.deviceAddressManager)
```

## f3e0153977fb
- project: `wifi_testapp`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-redundant-state-var` (suggestion)
- file: `entry/src/main/ets/pages/subBenchmarkTest/wifiManagerBenchmarkTest.ets`:81:3
- patch: shape=`modify`, changed_lines=31

### Before (local context)
```
    75    @State message: string = 'WifiManagerBenchmarkTest';
    76    private testItem: TestData = (router.getParams() as myParams).testItem
    77    @State changeIndex: number = - 1
    78    @StorageLink("wifiManagerBenchmarkTestMessage") wifiManagerBenchmarkTestMessage: string = ""
    79    @State w_ssid1: string = "TP-LINK_6365";
    80    @State w_bssid1: string = "6C:B1:58:75:63:65";
    81    @State w_preSharedKey1: string = "12345678";
    82    @State w_isHiddenSsid1: boolean = false;
    83    @State w_securityType1: number = 3
    84    @State w_creatorUid1: number = 1;
    85    @State w_disableReason1: number = 0;
    86    @State w_netId1: number = 0;
    87    @State w_randomMacType1: number = 0;
```

### After (local context)
```
    75    @State showList: boolean = false;
    76    message: string = 'WifiManagerBenchmarkTest';
    77    private testItem: TestData = (router.getParams() as myParams).testItem
    78    changeIndex: number = - 1
    79    @StorageLink("wifiManagerBenchmarkTestMessage") wifiManagerBenchmarkTestMessage: string = ""
    80    w_ssid1: string = "TP-LINK_6365";
    81    w_bssid1: string = "6C:B1:58:75:63:65";
    82    w_preSharedKey1: string = "12345678";
    83    w_isHiddenSsid1: boolean = false;
    84    w_securityType1: number = 3
    85    w_creatorUid1: number = 1;
    86    w_disableReason1: number = 0;
    87    w_netId1: number = 0;
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/subBenchmarkTest/wifiManagerBenchmarkTest.ets
+++ after/entry/src/main/ets/pages/subBenchmarkTest/wifiManagerBenchmarkTest.ets
@@ -1,25 +1,25 @@
+ *  WifiManager benchmark Test Page of wifi test
  */
 
 @Entry
 @Component
 struct WifiManagerBenchmarkTest {
   @State showList: boolean = false;
-  @State message: string = 'WifiManagerBenchmarkTest';
+  message: string = 'WifiManagerBenchmarkTest';
   private testItem: TestData = (router.getParams() as myParams).testItem
-  @State changeIndex: number = - 1
+  changeIndex: number = - 1
   @StorageLink("wifiManagerBenchmarkTestMessage") wifiManagerBenchmarkTestMessage: string = ""
-  @State w_ssid1: string = "TP-LINK_6365";
-  @State w_bssid1: string = "6C:B1:58:75:63:65";
-  @State w_preSharedKey1: string = "12345678";
-  @State w_isHiddenSsid1: boolean = false;
-  @State w_securityType1: number = 3
-  @State w_creatorUid1: number = 1;
-  @State w_disableReason1: number = 0;
-  @State w_netId1: number = 0;
-  @State w_randomMacType1: number = 0;
-  @State w_randomMacAddr1: string = "08:fb:ea:1b:38:aa"
-  @State w_ipType1: number = 1;
-  @State w_staticIp_ipAddress1: number = 3232235880;
-  @State w_staticIp_gateway1: number = 3232235777;
-  @State w_staticIp_dnsServers1: number = 3716386629;
-  @State w_staticIp_domains1: Array<string> = [ "0", "1", "2" ];
+  w_ssid1: string = "TP-LINK_6365";
+  w_bssid1: string = "6C:B1:58:75:63:65";
+  w_preSharedKey1: string = "12345678";
+  w_isHiddenSsid1: boolean = false;
+  w_securityType1: number = 3
+  w_creatorUid1: number = 1;
+  w_disableReason1: number = 0;
+  w_netId1: number = 0;
+  w_randomMacType1: number = 0;
+  w_randomMacAddr1: string = "08:fb:ea:1b:38:aa"
+  w_ipType1: number = 1;
+  w_staticIp_ipAddress1: number = 3232235880;
+  w_staticIp_gateway1: number = 3232235777;
+  w_staticIp_dnsServers1: number = 3716386629;
```

## 96aea5b2c5ec
- project: `wifi_testapp`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-redundant-state-var` (suggestion)
- file: `entry/src/main/ets/pages/subBenchmarkTest/wifiManagerBenchmarkTest.ets`:212:3
- patch: shape=`modify`, changed_lines=31

### Before (local context)
```
   206    }
   207  }
   208  
   209  @Component
   210  struct socketTest {
   211    @State login_feng: boolean = false
   212    @State login_wen: boolean = false
   213    @State user: string = ''
   214    @State roomDialog: boolean = false
   215    @State confirmDialog: boolean = false
   216    @State ipDialog: boolean = true
   217    @State txtDialog: boolean = true
   218    @State warnDialog: boolean = false
```

### After (local context)
```
   206      }
   207    }
   208  }
   209  
   210  @Component
   211  struct socketTest {
   212    login_feng: boolean = false
   213    login_wen: boolean = false
   214    user: string = ''
   215    roomDialog: boolean = false
   216    @State confirmDialog: boolean = false
   217    ipDialog: boolean = true
   218    @State txtDialog: boolean = true
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/subBenchmarkTest/wifiManagerBenchmarkTest.ets
+++ after/entry/src/main/ets/pages/subBenchmarkTest/wifiManagerBenchmarkTest.ets
@@ -1,3 +1,4 @@
+          }
         }
         .backgroundColor($r("sys.color.ohos_id_color_sub_background"))
         .width(ConfigData.WH_100_100)
@@ -9,17 +10,16 @@
 
 @Component
 struct socketTest {
-  @State login_feng: boolean = false
-  @State login_wen: boolean = false
-  @State user: string = ''
-  @State roomDialog: boolean = false
+  login_feng: boolean = false
+  login_wen: boolean = false
+  user: string = ''
+  roomDialog: boolean = false
   @State confirmDialog: boolean = false
-  @State ipDialog: boolean = true
+  ipDialog: boolean = true
   @State txtDialog: boolean = true
   @State warnDialog: boolean = false
   @State warnText: string = ''
-  @State roomNumber: string = ''
+  roomNumber: string = ''
   @State bindMsg: string = "未绑定"
   @State receiveMsg: string = '待接收数据'
 
-  bindOption() {
```

## dc9379440473
- project: `wifi_testapp`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-use-local-var-to-replace-state-var` (warn)
- file: `entry/src/main/ets/pages/subStabilityTest/hotspotStabilityTest.ets`:352:13
- patch: shape=`modify`, changed_lines=15

### Before (local context)
```
   346            console.log(TAG , "第" + this.closeHotspotNumber + "次热点去使能结果：" + funcMessage)
   347            console.log(TAG , "closeHotspotNumber: " + this.closeHotspotNumber)
   348            await sleep(10)
   349            this.hotspotMessage = AppStorage.get("hotspotMessage") ! //非空断言操作符
   350            if ( this.hotspotMessage == "inactive" ) {
   351              this.close_SpendTime = this.close_EndTime - this.close_StartTime
   352              this.hotspotMessageLog += "第" + this.closeHotspotNumber + "次热点去使能耗时: " + this.close_SpendTime + "ms" + "\n"
   353              console.log(TAG , "第" + this.closeHotspotNumber + "次热点去使能耗时: " + this.close_SpendTime + "ms")
   354              this.closeSuccessNumber = this.closeSuccessNumber + 1
   355              this.hotspotMessageLog += "热点去使能成功的次数：" + this.closeSuccessNumber + "\n"
   356              console.log(TAG , "热点去使能成功的次数：" + this.closeSuccessNumber)
   357              await sleep(7)
   358            } else {
```

### After (local context)
```
   346            await sleep(10)
   347            this.hotspotMessage = AppStorage.get("hotspotMessage") ! //非空断言操作符
   348            if ( this.hotspotMessage == "inactive" ) {
   349              this.close_SpendTime = this.close_EndTime - this.close_StartTime
   350              hotspotMessageLog += "第" + this.closeHotspotNumber + "次热点去使能耗时: " + this.close_SpendTime + "ms" + "\n"
   351              console.log(TAG , "第" + this.closeHotspotNumber + "次热点去使能耗时: " + this.close_SpendTime + "ms")
   352              this.closeSuccessNumber = this.closeSuccessNumber + 1
   353              hotspotMessageLog += "热点去使能成功的次数：" + this.closeSuccessNumber + "\n"
   354              console.log(TAG , "热点去使能成功的次数：" + this.closeSuccessNumber)
   355              await sleep(7)
   356            } else {
   357              this.closeFailNumber = this.closeFailNumber + 1
   358              console.log(TAG , "热点去使能失败的次数：" + this.closeFailNumber)
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/subStabilityTest/hotspotStabilityTest.ets
+++ after/entry/src/main/ets/pages/subStabilityTest/hotspotStabilityTest.ets
@@ -1,19 +1,17 @@
-          // funcMessage = wifiManager.disableHotspot()
-          this.closeHotspotNumber = this.closeHotspotNumber + 1
           this.close_StartTime = new Date().getTime()
           console.log(TAG , "第" + this.closeHotspotNumber + "次热点去使能-----")
           console.log(TAG , "第" + this.closeHotspotNumber + "次热点去使能开始时间: " + this.close_StartTime + "ms")
-          this.hotspotMessageLog += "第" + this.closeHotspotNumber + "次热点去使能结果：" + funcMessage + "\n"
+          hotspotMessageLog += "第" + this.closeHotspotNumber + "次热点去使能结果：" + funcMessage + "\n"
           console.log(TAG , "第" + this.closeHotspotNumber + "次热点去使能结果：" + funcMessage)
           console.log(TAG , "closeHotspotNumber: " + this.closeHotspotNumber)
           await sleep(10)
           this.hotspotMessage = AppStorage.get("hotspotMessage") ! //非空断言操作符
           if ( this.hotspotMessage == "inactive" ) {
             this.close_SpendTime = this.close_EndTime - this.close_StartTime
-            this.hotspotMessageLog += "第" + this.closeHotspotNumber + "次热点去使能耗时: " + this.close_SpendTime + "ms" + "\n"
+            hotspotMessageLog += "第" + this.closeHotspotNumber + "次热点去使能耗时: " + this.close_SpendTime + "ms" + "\n"
             console.log(TAG , "第" + this.closeHotspotNumber + "次热点去使能耗时: " + this.close_SpendTime + "ms")
             this.closeSuccessNumber = this.closeSuccessNumber + 1
-            this.hotspotMessageLog += "热点去使能成功的次数：" + this.closeSuccessNumber + "\n"
+            hotspotMessageLog += "热点去使能成功的次数：" + this.closeSuccessNumber + "\n"
             console.log(TAG , "热点去使能成功的次数：" + this.closeSuccessNumber)
             await sleep(7)
           } else {
@@ -23,3 +21,5 @@
           }
         } else if ( this.hotspotMessage == "inactive" ) {
           this.openFailNumber = this.openFailNumber + 1
+          console.log(TAG , "热点使能失败的次数：" + this.openFailNumber)
+          console.log(TAG , "第" + (this.openHotspotNumber + 1) + "次热点使能失败")
```

## 805cd1d62a1b
- project: `wifi_testapp`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-unchanged-state-var` (suggestion)
- file: `entry/src/main/ets/pages/subStabilityTest/wifiSwitchStabilityTest.ets`:74:3
- patch: shape=`modify`, changed_lines=29

### Before (local context)
```
    68    @State openFailNumber: number = 0 // 打开WiFi的失败次数
    69    @State closeSuccessNumber: number = 0 // 关闭WiFi的成功次数
    70    @State closeFailNumber: number = 0 // 关闭WiFi的失败次数
    71    @State message: string = "测试结果:"
    72    @State testNumbers: number = 30 //测试次数
    73    @State successTimes: number = 0
    74    @State failTimes: number = 0
    75    @State stateMessage: string = ""
    76    @State stateMessageLog: string = ""
    77    @State switchLoopState: Boolean = true
    78    @State fileData: string = "";
    79    @State filePath: string = ""
    80    private file!: fs.File
```

### After (local context)
```
    68    @State openSuccessNumber: number = 0 // 打开WiFi的成功次数
    69    @State openFailNumber: number = 0 // 打开WiFi的失败次数
    70    @State closeSuccessNumber: number = 0 // 关闭WiFi的成功次数
    71    @State closeFailNumber: number = 0 // 关闭WiFi的失败次数
    72    message: string = "测试结果:"
    73    @State testNumbers: number = 30 //测试次数
    74    successTimes: number = 0
    75    failTimes: number = 0
    76    @State stateMessage: string = ""
    77    @State stateMessageLog: string = ""
    78    @State switchLoopState: Boolean = true
    79    fileData: string = "";
    80    @State filePath: string = ""
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/subStabilityTest/wifiSwitchStabilityTest.ets
+++ after/entry/src/main/ets/pages/subStabilityTest/wifiSwitchStabilityTest.ets
@@ -1,3 +1,4 @@
+  @State open_SpendTime: number = 0
   @State close_StartTime: number = 0
   @State close_EndTime: number = 0
   @State close_SpendTime: number = 0
@@ -7,14 +8,14 @@
   @State openFailNumber: number = 0 // 打开WiFi的失败次数
   @State closeSuccessNumber: number = 0 // 关闭WiFi的成功次数
   @State closeFailNumber: number = 0 // 关闭WiFi的失败次数
-  @State message: string = "测试结果:"
+  message: string = "测试结果:"
   @State testNumbers: number = 30 //测试次数
-  @State successTimes: number = 0
-  @State failTimes: number = 0
+  successTimes: number = 0
+  failTimes: number = 0
   @State stateMessage: string = ""
   @State stateMessageLog: string = ""
   @State switchLoopState: Boolean = true
-  @State fileData: string = "";
+  fileData: string = "";
   @State filePath: string = ""
   private file!: fs.File
   @StorageLink("pathDir") pathDir: string = ""
@@ -22,4 +23,3 @@
   async openLogFile() {
     let time: number = new Date().getTime()
     let currentTime: string = timestampToDate(time)
-    console.log(TAG , "当前时间: " + currentTime)
```

## d0b922149766
- project: `wifi_testapp`
- round_fixed: `2` (early)
- rule: `performance/hp-arkui-use-local-var-to-replace-state-var` (warn)
- file: `entry/src/main/ets/pages/subAppTest/rcpHttpTest.ets`:376:7
- patch: shape=`modify`, changed_lines=25

### Before (local context)
```
   370  
   371        const endTime = Date.now();
   372        const duration = endTime - startTime;
   373  
   374        // 处理响应
   375        this.isLoading = false;
   376        this.codeValue = response.statusCode.toString();
   377  
   378        // 构建响应数据显示
   379        let resData = this.resData;
   380        resData = `请求方法: ${this.selectValue}\n`;
   381        resData += `请求URL: ${this.urlValue}\n`;
   382        resData += `\n=== 响应信息 ===\n`;
```

### After (local context)
```
   370            response = await this.rcpSession.head(this.urlValue);
   371            break;
   372          case 'FETCH':
   373            const request = new rcp.Request(this.urlValue, 'POST');
   374            response = await this.rcpSession.fetch(request);
   375            break;
   376          default:
   377            response = await this.rcpSession.get(this.urlValue);
   378        }
   379  
   380        const endTime = Date.now();
   381        const duration = endTime - startTime;
   382  
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/subAppTest/rcpHttpTest.ets
+++ after/entry/src/main/ets/pages/subAppTest/rcpHttpTest.ets
@@ -1,3 +1,12 @@
+          response = await this.rcpSession.put(this.urlValue, this.reqValue);
+          break;
+        case 'DELETE':
+          response = await this.rcpSession.delete(this.urlValue);
+          break;
+        case 'HEAD':
+          response = await this.rcpSession.head(this.urlValue);
+          break;
+        case 'FETCH':
           const request = new rcp.Request(this.urlValue, 'POST');
           response = await this.rcpSession.fetch(request);
           break;
@@ -9,17 +18,8 @@
       const duration = endTime - startTime;
 
       // 处理响应
-      this.isLoading = false;
-      this.codeValue = response.statusCode.toString();
-
-      // 构建响应数据显示
-      let resData = this.resData;
-      resData = `请求方法: ${this.selectValue}\n`;
-      resData += `请求URL: ${this.urlValue}\n`;
-      resData += `\n=== 响应信息 ===\n`;
-      resData += `状态码: ${response.statusCode}\n`;
-      resData += `响应时间: ${duration}ms\n`;
-
-      // 显示响应头
-      if (response.headers) {
-        resData += `\n=== 响应头 ===\n`;
+      let isLoading = this.isLoading;
+      isLoading = false;
+      this.isLoading = isLoading;
+      let codeValue = this.codeValue;
+      codeValue = response.statusCode.toString();
```

## 54b10764d5ae
- project: `wifi_testapp`
- round_fixed: `2` (early)
- rule: `performance/foreach-index-check` (suggestion)
- file: `entry/src/main/ets/pages/subBenchmarkTest/hotspotBenchmarkTest.ets`:396:11
- patch: shape=`add`, changed_lines=1

### Before (local context)
```
   390        Scroll() {
   391          Column() {
   392            Text($r('app.string.choice_download_file'))
   393              .fontSize(25)
   394              .alignSelf(ItemAlign.Start)
   395              .margin({ top : 20 , left : 10 })
   396            ForEach(this.files , (item: string , index) => {
   397              Divider()
   398                .margin({ top : 20 })
   399              Row() {
   400                Text(item)
   401                  .fontSize(25)
   402                  .constraintSize({ maxWidth : ConfigData.WH_75_100 })
```

### After (local context)
```
   390      Column() {
   391        Scroll() {
   392          Column() {
   393            Text($r('app.string.choice_download_file'))
   394              .fontSize(25)
   395              .alignSelf(ItemAlign.Start)
   396              .margin({ top : 20 , left : 10 })
   397            ForEach(this.files , (item: string , index) => {
   398              Divider()
   399                .margin({ top : 20 })
   400              Row() {
   401                Text(item)
   402                  .fontSize(25)
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/subBenchmarkTest/hotspotBenchmarkTest.ets
+++ after/entry/src/main/ets/pages/subBenchmarkTest/hotspotBenchmarkTest.ets
@@ -1,3 +1,4 @@
+        httpRequest.destroy()
       }
     })
   }
@@ -22,4 +23,3 @@
               Blank()
 
               Button($r('app.string.click_download'))
-                .margin({ top : 20 , right : 10 })
```

## a13c8ad0d20f
- project: `CanvasTest`
- round_fixed: `1` (early)
- rule: `security/no-commented-code` (warn)
- file: `entry/src/main/ets/pages/CanvasPage.ets`:282:5
- patch: shape=`modify`, changed_lines=11

### Before (local context)
```
   276      }
   277  
   278      return this.rootNode;
   279    }
   280  
   281    notifyRedraw() {
   282      // this.canvasNode?.notifyRedraw();
   283      // this.canvasNode.invalidate();
   284      if (!this.nodeIsDrawing) {
   285        this.nodeIsDrawing = true;
   286        this.canvasNode?.notifyRedraw();
   287        this.uiContext.postFrameCallback(new RenderFrameCallback(this));
   288      }
```

### After (local context)
```
   276  
   277    notifyRedraw() {
   278      if (!this.nodeIsDrawing) {
   279        this.nodeIsDrawing = true;
   280        this.canvasNode?.notifyRedraw();
   281        this.uiContext.postFrameCallback(new RenderFrameCallback(this));
   282      }
   283    }
   284  
   285    onTouchEvent(event: TouchEvent): void {
   286      let x: number = 0;
   287      let y: number = 0;
   288      let id: number = 0;
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/CanvasPage.ets
+++ after/entry/src/main/ets/pages/CanvasPage.ets
@@ -1,7 +1,3 @@
-      // if (!this.canvasNode) {
-      this.canvasNode = nativeCanvas.nativeCreateRootNode() as CanvasRenderNode;
-      // }
-      rootRenderNode.appendChild(this.canvasNode);
       hilog.info(0x0000, 'testTag', `rootNode id: ${this.rootNode.getId()}`);
       hilog.info(0x0000, 'testTag', `add root node ${this.canvasNode}}`);
     }
@@ -10,8 +6,6 @@
   }
 
   notifyRedraw() {
-    // this.canvasNode?.notifyRedraw();
-    // this.canvasNode.invalidate();
     if (!this.nodeIsDrawing) {
       this.nodeIsDrawing = true;
       this.canvasNode?.notifyRedraw();
@@ -23,3 +17,9 @@
     let x: number = 0;
     let y: number = 0;
     let id: number = 0;
+    let changedTouchLen = event.changedTouches.length;
+    for (let i = 0; i < changedTouchLen; i++) {
+      x = vp2px(event.changedTouches[i].x);
+      y = vp2px(event.changedTouches[i].y);
+      id = event.changedTouches[i].id;
+      let moveType: number = 0;
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

## b7eb2c597d85
- project: `audio_suite`
- round_fixed: `1` (early)
- rule: `security/no-commented-code` (warn)
- file: `entry/src/main/ets/pages/AutoTest.ets`:890:3
- patch: shape=`modify`, changed_lines=41

### Before (local context)
```
   884    //       await this.saveSingleFile(pipelineId, buffer as ArrayBuffer[], ".wav", outputInfo, multiRenderFrameFlag)
   885    //       this.generateSingleReportInfo(pipelineId);
   886    //     })
   887    //     await Promise.all(scenePromises).then(async () => {
   888    //       Logger.info(TAG, `scenePromises done `);
   889    //       if (reportFlag) {
   890    //         await this.saveSingleCaseExecuteReport(fileSummary);
   891    //       }
   892    //       this.destroyPipeline();
   893    //     })
   894    //   }
   895    //   return 0;
   896    // }
```

### After (local context)
```
   884      if (jsonFileInfo == undefined) {
   885        return 1;
   886      } else if (jsonFileInfo.sceneInfos.length >= 1) {
   887        let status = this.initAllPipeline(jsonFileInfo);
   888        if (status != SUCCESS) {
   889          Logger.error(TAG, `initAllPipeline ERROR`);
   890          return FAILED;
   891        }
   892        let taskExeStatus = await taskpool.execute(initAllInputNode, jsonFileInfo, sceneInfoIdToSceneInfoMap, pipelineIdToPipelineInfoMap);
   893        if (taskExeStatus != SUCCESS) {
   894          Logger.error(TAG, `initAllInputNode ERROR`);
   895          return FAILED;
   896        }
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/AutoTest.ets
+++ after/entry/src/main/ets/pages/AutoTest.ets
@@ -1,20 +1,3 @@
-  //       let ret = await this.initMultiPipelineTask(sceneInfo.sceneInfoId, pipelineId);
-  //       Logger.info(TAG, `initMultiPipelineTask done pipelineId：${pipelineId}`);
-  //       let outputInfo = sceneInfo.outputFile;
-  //       const multiRenderFrameFlag = pipelineInfo.multiRenderFrameFlag;
-  //       const buffer: object =
-  //         await taskpool.execute(multiPipelineSaveFileBuffer, pipelineId, outputInfo, multiRenderFrameFlag);
-  //       await this.saveSingleFile(pipelineId, buffer as ArrayBuffer[], ".wav", outputInfo, multiRenderFrameFlag)
-  //       this.generateSingleReportInfo(pipelineId);
-  //     })
-  //     await Promise.all(scenePromises).then(async () => {
-  //       Logger.info(TAG, `scenePromises done `);
-  //       if (reportFlag) {
-  //         await this.saveSingleCaseExecuteReport(fileSummary);
-  //       }
-  //       this.destroyPipeline();
-  //     })
-  //   }
   //   return 0;
   // }
 
@@ -23,3 +6,20 @@
     const jsonFileInfo = jsonSummaryToJsonFileInfoMap.get(fileSummary);
     if (jsonFileInfo == undefined) {
       return 1;
+    } else if (jsonFileInfo.sceneInfos.length >= 1) {
+      let status = this.initAllPipeline(jsonFileInfo);
+      if (status != SUCCESS) {
+        Logger.error(TAG, `initAllPipeline ERROR`);
+        return FAILED;
+      }
+      let taskExeStatus = await taskpool.execute(initAllInputNode, jsonFileInfo, sceneInfoIdToSceneInfoMap, pipelineIdToPipelineInfoMap);
+      if (taskExeStatus != SUCCESS) {
+        Logger.error(TAG, `initAllInputNode ERROR`);
+        return FAILED;
+      }
+      status = this.initAllEffectNode(jsonFileInfo);
+      if (status != SUCCESS) {
+        Logger.error(TAG, `initAllEffectNode ERROR`);
+        return FAILED;
+      }
+      Logger.info(TAG, `initAllEffectNode successed, ready to render frame`);
```

## a28f9341dafb
- project: `flutter_embedding`
- round_fixed: `1` (early)
- rule: `security/no-commented-code` (warn)
- file: `flutter/src/main/ets/embedding/ohos/KeyboardMap.ets`:92:39
- patch: shape=`modify`, changed_lines=4

### Before (local context)
```
    86        [0x00000000809, 0x0000000002D], // minus
    87        [0x0000000080A, 0x0000000003D], // equal
    88        [0x0000000080B, 0x0000000005B], // bracketLeft
    89        [0x0000000080C, 0x0000000005D], // bracketRight
    90        [0x0000000080D, 0x0000000005C], // backslash
    91        [0x0000000080E, 0x0000000003B], // semicolon
    92        [0x0000000080F, 0x00000000022], // apostrophe/quote
    93        [0x00000000810, 0x0000000002F], // slash
    94        [0x00000000813, 0x00100000505], // contextMenu
    95        [0x000000009A2, 0x00100000704], // compose
    96        [0x00000000814, 0x00100000308], // pageUp
    97        [0x00000000815, 0x00100000307], // pageDown
    98        [0x00000000816, 0x0010000001B], // escape
```

### After (local context)
```
    86        [0x00000000808, 0x00000000060], // backquote
    87        [0x00000000809, 0x0000000002D], // minus
    88        [0x0000000080A, 0x0000000003D], // equal
    89        [0x0000000080B, 0x0000000005B], // bracketLeft
    90        [0x0000000080C, 0x0000000005D], // bracketRight
    91        [0x0000000080D, 0x0000000005C], // backslash
    92        [0x0000000080E, 0x0000000003B], // semicolon
    93        [0x00000000810, 0x0000000002F], // slash
    94        [0x00000000813, 0x00100000505], // contextMenu
    95        [0x000000009A2, 0x00100000704], // compose
    96        [0x00000000814, 0x00100000308], // pageUp
    97        [0x00000000815, 0x00100000307], // pageDown
    98        [0x00000000816, 0x0010000001B], // escape
```

### Local diff
```diff
--- before/flutter/src/main/ets/embedding/ohos/KeyboardMap.ets
+++ after/flutter/src/main/ets/embedding/ohos/KeyboardMap.ets
@@ -1,3 +1,4 @@
+      [0x00000000801, 0x00100000009], // tab
       [0x00000000802, 0x00000000020], // space
       [0x00000000804, 0x00100000B09], // launchWebBrowser
       [0x00000000805, 0x00100000B03], // launchMail
@@ -10,7 +11,6 @@
       [0x0000000080C, 0x0000000005D], // bracketRight
       [0x0000000080D, 0x0000000005C], // backslash
       [0x0000000080E, 0x0000000003B], // semicolon
-      [0x0000000080F, 0x00000000022], // apostrophe/quote
       [0x00000000810, 0x0000000002F], // slash
       [0x00000000813, 0x00100000505], // contextMenu
       [0x000000009A2, 0x00100000704], // compose
```

## 8696e87c007b
- project: `ohos_cordova`
- round_fixed: `1` (early)
- rule: `security/no-cycle` (warn)
- file: `library/src/main/ets/cordova/CordovaBridge.ets`:2:1
- patch: shape=`modify`, changed_lines=2

### Before (local context)
```
     1  import cryptoFramework from '@ohos.security.cryptoFramework'
     2  import { NativeToJsMessageQueue } from './NativeToJsMessageQueue'
     3  import { PluginManager } from './PluginManager'
     4  import { NumberUtils, StringUtils } from '../utils/Index'
     5  
     6  /**
     7   * Cordova原生和H5的通信桥
     8   *
```

### After (local context)
```
     1  
     2  import cryptoFramework from '@ohos.security.cryptoFramework'
     3  import { NativeToJsMessageQueue } from './Node3.ets'
     4  import { PluginManager } from './PluginManager'
     5  import { NumberUtils, StringUtils } from '../utils/Index'
     6  
     7  /**
     8   * Cordova原生和H5的通信桥
```

### Local diff
```diff
--- before/library/src/main/ets/cordova/CordovaBridge.ets
+++ after/library/src/main/ets/cordova/CordovaBridge.ets
@@ -1,5 +1,6 @@
+
 import cryptoFramework from '@ohos.security.cryptoFramework'
-import { NativeToJsMessageQueue } from './NativeToJsMessageQueue'
+import { NativeToJsMessageQueue } from './Node3.ets'
 import { PluginManager } from './PluginManager'
 import { NumberUtils, StringUtils } from '../utils/Index'
 
@@ -11,4 +12,3 @@
  * @date 2023/12/26 09:09
  */
 export class CordovaBridge {
-  private readonly pluginManager: PluginManager
```

## 759d8f05cfc5
- project: `ohos_dfu_library`
- round_fixed: `1` (early)
- rule: `security/no-commented-code` (warn)
- file: `DfuLibrary/src/main/ets/dfu/BaseDfuImpl.ts`:325:5
- patch: shape=`modify`, changed_lines=18

### Before (local context)
```
   319  
   320      // TODO: need restart service
   321      // if (newAddress != null)
   322      // intent.putExtra(DfuBaseService.EXTRA_DEVICE_ADDRESS, newAddress);
   323      //
   324      // // Reset the DFU attempt counter
   325      // intent.putExtra(DfuBaseService.EXTRA_DFU_ATTEMPT, 0);
   326      //
   327      // final boolean foregroundService = intent.getBooleanExtra(DfuBaseService.EXTRA_FOREGROUND_SERVICE, true);
   328      // if (foregroundService && Build.VERSION.SDK_INT >= Build.VERSION_CODES.O)
   329      // mService.startForegroundService(intent);
   330      // else
   331      // mService.startService(intent);
```

### After (local context)
```
   317  
   318    }
   319  
   320    protected abstract getDfuServiceUUID(): string;
   321  
   322  
   323  }
```

### Local diff
```diff
--- before/DfuLibrary/src/main/ets/dfu/BaseDfuImpl.ts
+++ after/DfuLibrary/src/main/ets/dfu/BaseDfuImpl.ts
@@ -1,25 +1,11 @@
-        hilog.info(this.DOMAIN, this.TAG, "DFU Bootloader found with address " + newAddress);
-      else {
-        // mService.sendLogBroadcast(DfuBaseService.LOG_LEVEL_INFO, "DFU Bootloader not found. Trying the same address...");
-        hilog.info(this.DOMAIN, this.TAG, "DFU Bootloader not found. Trying the same address...");
-      }
-    }
-
-    // TODO: need restart service
-    // if (newAddress != null)
-    // intent.putExtra(DfuBaseService.EXTRA_DEVICE_ADDRESS, newAddress);
-    //
-    // // Reset the DFU attempt counter
-    // intent.putExtra(DfuBaseService.EXTRA_DFU_ATTEMPT, 0);
     //
     // final boolean foregroundService = intent.getBooleanExtra(DfuBaseService.EXTRA_FOREGROUND_SERVICE, true);
     // if (foregroundService && Build.VERSION.SDK_INT >= Build.VERSION_CODES.O)
-    // mService.startForegroundService(intent);
     // else
-    // mService.startService(intent);
 
   }
 
   protected abstract getDfuServiceUUID(): string;
 
 
+}
```

## 1271545761ee
- project: `ohos_dfu_library`
- round_fixed: `1` (early)
- rule: `security/no-commented-code` (warn)
- file: `entry/src/main/ets/pages/Index.ets`:258:9
- patch: shape=`modify`, changed_lines=32

### Before (local context)
```
   252                console.info(`scanres: ${item.deviceName},${item.deviceId}`);
   253              }
   254            }
   255          });
   256  
   257          let scanFilter: ble.ScanFilter = {};
   258          // let scanFilter: ble.ScanFilter = {name:'MD2-HA20-F220231200085'};
   259          // let scanFilter: ble.ScanFilter = {
   260          //   deviceId:"XX:XX:XX:XX:XX:XX",
   261          //   name:"test",
   262          //   serviceUuid:"00001888-0000-1000-8000-00805f9b34fb"
   263          // };
   264          let scanOptions: ble.ScanOptions = {
```

### After (local context)
```
   252                console.info(`scanres: ${item.deviceName},${item.deviceId}`);
   253              }
   254            }
   255          });
   256  
   257          let scanFilter: ble.ScanFilter = {};
   258          let scanOptions: ble.ScanOptions = {
   259            interval: 500,
   260            dutyMode: ble.ScanDuty.SCAN_MODE_LOW_POWER,
   261            matchMode: ble.MatchMode.MATCH_MODE_AGGRESSIVE,
   262          }
   263          ble.startBLEScan([scanFilter],scanOptions);
   264        } catch (err) {
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/Index.ets
+++ after/entry/src/main/ets/pages/Index.ets
@@ -10,12 +10,6 @@
         });
 
         let scanFilter: ble.ScanFilter = {};
-        // let scanFilter: ble.ScanFilter = {name:'MD2-HA20-F220231200085'};
-        // let scanFilter: ble.ScanFilter = {
-        //   deviceId:"XX:XX:XX:XX:XX:XX",
-        //   name:"test",
-        //   serviceUuid:"00001888-0000-1000-8000-00805f9b34fb"
-        // };
         let scanOptions: ble.ScanOptions = {
           interval: 500,
           dutyMode: ble.ScanDuty.SCAN_MODE_LOW_POWER,
@@ -23,3 +17,9 @@
         }
         ble.startBLEScan([scanFilter],scanOptions);
       } catch (err) {
+        hilog.error(DOMAIN, TAG, 'errCode: ' + (err as BusinessError).code + ', errMessage: ' + (err as BusinessError).message);
+      }
+      return true;
+    }
+    return false;
+  }
```

## b90f3c5f6133
- project: `ohos_mail_base`
- round_fixed: `1` (early)
- rule: `security/no-commented-code` (warn)
- file: `test/src/main/ets/pages/MailDetail.ets`:49:7
- patch: shape=`modify`, changed_lines=19

### Before (local context)
```
    43          Text("time：")
    44          Text(this.itemMailInfo.time)
    45        }
    46  
    47        // Row({ space: 20 }) {
    48        //   Text("subject：")
    49        //   Text(this.itemMailInfo.subject)
    50        // }
    51        //
    52        // Row({ space: 20 }) {
    53        //   Text("date：")
    54        //   Text(this.itemMailInfo.date)
    55        // }
```

### After (local context)
```
    43  
    44        // Row({ space: 20 }) {
    45        // }
    46        //
    47        // Row({ space: 20 }) {
    48        // }
    49        //
    50        // Row({ space: 20 }) {
    51        //   Text("是否有附件：")
    52        //   Text(`${this.itemMailInfo.hasAttach ? '有' : '无'}`)
    53        // }
    54        //
    55        // Row({ space: 20 }) {
```

### Local diff
```diff
--- before/test/src/main/ets/pages/MailDetail.ets
+++ after/test/src/main/ets/pages/MailDetail.ets
@@ -1,21 +1,14 @@
-      Row({ space: 20 }) {
-        Text("digest：")
-        Text(this.itemMailInfo.digest)
       }
 
       Row({ space: 20 }) {
         Text("time：")
-        Text(this.itemMailInfo.time)
+        Text(itemMailInfo.time)
       }
 
       // Row({ space: 20 }) {
-      //   Text("subject：")
-      //   Text(this.itemMailInfo.subject)
       // }
       //
       // Row({ space: 20 }) {
-      //   Text("date：")
-      //   Text(this.itemMailInfo.date)
       // }
       //
       // Row({ space: 20 }) {
@@ -23,3 +16,10 @@
       //   Text(`${this.itemMailInfo.hasAttach ? '有' : '无'}`)
       // }
       //
+      // Row({ space: 20 }) {
+      // }
+      //
+      // Row({ space: 20 }) {
+      // }
+    }
+    .alignItems(HorizontalAlign.Start)
```

## 7a0c4cac8f75
- project: `Image`
- round_fixed: `4` (late)
- rule: `performance/hp-arkui-no-state-var-access-in-loop` (warn)
- file: `entry/src/main/ets/pages/ImageParallelization/ImageCallBack.ets`:502:21
- patch: shape=`modify`, changed_lines=43

### Before (local context)
```
   496                  Image($r('app.media.startIcon'))
   497                    .width(this.imageWidth)
   498                    .height(200)
   499                    .objectFit(ImageFit.Contain)
   500                    .visibility(this.visible)
   501                    .onComplete(() => {
   502                      this.visible = Visibility.Visible;
   503                      this.moveImg.pop();
   504                      console.info('Test onComplete')
   505                    })
   506                    .onError(() => {
   507                      setTimeout(() => {
   508                        this.visible = Visibility.Visible;
```

### After (local context)
```
   496                  Text('---------------------------').fontSize(14)
   497                  Text('事件').fontSize(22).margin({ bottom: 15 } as Margin)
   498  
   499                  Image($r('app.media.startIcon'))
   500                    .width(this.imageWidth)
   501                    .height(200)
   502                    .objectFit(ImageFit.Contain)
   503                    .visibility(this.visible)
   504                    .onComplete(() => {
   505                      this.visible = Visibility.Visible;
   506                      this.moveImg.pop();
   507                      console.info('Test onComplete')
   508                    })
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/ImageParallelization/ImageCallBack.ets
+++ after/entry/src/main/ets/pages/ImageParallelization/ImageCallBack.ets
@@ -1,3 +1,6 @@
+            })
+          }
+          ParallelizeUI({}as ParallelOption) {
             ForEach(forEachItems100,(item: Int, index:Double)=>{
               Column(undefined) {
                 Text("image").fontSize(40)
@@ -20,6 +23,3 @@
                       this.moveImg.pop();
                       console.info('Test onError')
                     }, 2600)
-                  })
-                Image(this.src2)
-                  .width(100)
```

## 5c6d5eef7781
- project: `Image`
- round_fixed: `4` (late)
- rule: `performance/hp-arkui-no-state-var-access-in-loop` (warn)
- file: `entry/src/main/ets/pages/ImageParallelization/ImageCallBack.ets`:547:17
- patch: shape=`modify`, changed_lines=43

### Before (local context)
```
   541                      setTimeout(() => {
   542                        this.visible = Visibility.Visible;
   543                        this.moveImg.pop();
   544                        console.info('Test onError')
   545                      }, 2600)
   546                    })
   547                  Image(this.src2)
   548                    .width(100)
   549                    .height(100)
   550                    .onFinish(() => {
   551                      this.src2 = this.imageOne;
   552                      console.info('Test onFinish')
   553                    })
```

### After (local context)
```
   531        }
   532      }
   533      this.this.this.visible = this.this.visible;
   534      this.this.this.src2 = this.this.src2;
   535      this.this.this.moveImg = this.this.moveImg;
   536    }
   537  }
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/ImageParallelization/ImageCallBack.ets
+++ after/entry/src/main/ets/pages/ImageParallelization/ImageCallBack.ets
@@ -1,25 +1,3 @@
-                  .onComplete(() => {
-                    this.visible = Visibility.Visible;
-                    this.moveImg.pop();
-                    console.info('Test onComplete')
-                  })
-                  .onError(() => {
-                    setTimeout(() => {
-                      this.visible = Visibility.Visible;
-                      this.moveImg.pop();
-                      console.info('Test onError')
-                    }, 2600)
-                  })
-                Image(this.src2)
-                  .width(100)
-                  .height(100)
-                  .onFinish(() => {
-                    this.src2 = this.imageOne;
-                    console.info('Test onFinish')
-                  })
-              }
-            })
-          }
-
-        }
-      }
+    this.this.this.moveImg = this.this.moveImg;
+  }
+}
```

## 31b088665eed
- project: `Info`
- round_fixed: `4` (late)
- rule: `performance/avoid-overusing-custom-component-check` (warn)
- file: `entry/src/main/ets/pages/badge/MyComponent.ets`:4:15
- patch: shape=`add`, changed_lines=1

### Before (local context)
```
     1  
     2  @Preview
     3  @Component
     4  export struct MyComponent {
     5    private title: string = 'test'
     6    private func: () => void = () => {
     7    }
     8  
     9    @Styles
    10    pressedStyle() {
```

### After (local context)
```
     1  
     2  
     3  @Preview
     4  @Component
     5  export struct MyComponent {
     6    private title: string = 'test'
     7    private func: () => void = () => {
     8    }
     9  
    10    @Styles
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/badge/MyComponent.ets
+++ after/entry/src/main/ets/pages/badge/MyComponent.ets
@@ -1,3 +1,4 @@
+
 
 @Preview
 @Component
@@ -13,4 +14,3 @@
 
   @Styles
   normalStyles() {
-    .backgroundColor(0x0000FF)
```

## dbd690e7811e
- project: `ace_ets_module_swiper_api11`
- round_fixed: `3` (late)
- rule: `performance/hp-arkui-no-func-as-arg-for-reusable-component` (warn)
- file: `entry/src/main/ets/MainAbility/pages/Swiper/Swiper/swiper1.ets`:77:11
- patch: shape=`delete`, changed_lines=2

### Before (local context)
```
    71    }
    72  
    73    build() {
    74      Column({ space: 5 }) {
    75        Swiper(this.swiperController) {
    76          LazyForEach(this.data, (item: string) => {
    77            SwiperItemComponent({
    78              item: item.toString()
    79            })
    80          }, (item: string) => item)
    81        }
    82        .key('swiperTest1')
    83        .cachedCount(2)
```

### After (local context)
```
    71    build() {
    72      Column({ space: 5 }) {
    73        Swiper(this.swiperController) {
    74          LazyForEach(this.data, (item: string) => {
    75            SwiperItemComponent({
    76              item: item.toString()
    77            })
    78          }, (item: string) => item)
    79        }
    80        .key('swiperTest1')
    81        .cachedCount(2)
    82        .index(1)
    83        .autoPlay(false)
```

### Local diff
```diff
--- before/entry/src/main/ets/MainAbility/pages/Swiper/Swiper/swiper1.ets
+++ after/entry/src/main/ets/MainAbility/pages/Swiper/Swiper/swiper1.ets
@@ -1,5 +1,3 @@
-    let list: number[] = []
-    for (let i = 1; i <= 10; i++) {
       list.push(i);
     }
     this.photos = list
@@ -23,3 +21,5 @@
       .vertical(true)
       .loop(true)
       .duration(1000)
+      .itemSpace(0)
+      .indicator( // 设置圆点导航点样式
```

## dc327bdebd03
- project: `applications_photos`
- round_fixed: `4` (late)
- rule: `performance/hp-arkui-no-func-as-arg-for-reusable-component` (warn)
- file: `product/phone/src/main/ets/pages/NewAlbumPage.ets`:403:19
- patch: shape=`add`, changed_lines=1

### Before (local context)
```
   397            })
   398          } else {
   399            Stack() {
   400              Grid(this.scroller, this.layoutOptions) {
   401                LazyForEach(this.dataSource, (item: ViewData, index?: number) => {
   402                  if (!!item) {
   403                    NewAlbumGridItemComponent({
   404                      dataSource: this.dataSource,
   405                      item: item.mediaItem,
   406                      isSelected: this.isSelectedMode ?
   407                      this.mSelectManager?.isItemSelected((item.mediaItem as MediaItem).uri as string, item.viewIndex) :
   408                        false,
   409                      pageName: Constants.PHOTO_TRANSITION_ALBUM,
```

### After (local context)
```
   397              title: $r('app.string.no_distributed_photo_head_title_album')
   398            })
   399          } else {
   400            Stack() {
   401              Grid(this.scroller, this.layoutOptions) {
   402                LazyForEach(this.dataSource, (item: ViewData, index?: number) => {
   403                  if (!!item) {
   404                    NewAlbumGridItemComponent({
   405                      dataSource: this.dataSource,
   406                      item: item.mediaItem,
   407                      isSelected: this.isSelectedMode ?
   408                      this.mSelectManager?.isItemSelected((item.mediaItem as MediaItem).uri as string, item.viewIndex) :
   409                        false,
```

### Local diff
```diff
--- before/product/phone/src/main/ets/pages/NewAlbumPage.ets
+++ after/product/phone/src/main/ets/pages/NewAlbumPage.ets
@@ -1,3 +1,4 @@
+          onMenuClicked: (action: Action): void => this.onMenuClicked(action),
           totalSelectedCount: $totalSelectedCount,
           menuList: $moreMenuList
         })
@@ -22,4 +23,3 @@
                     selectedCount: this.totalSelectedCount,
                     placeholderIndex: this.placeholderIndex
                   })
-                }
```

## bb4719285aa7
- project: `ohos_mail_base`
- round_fixed: `5` (late)
- rule: `performance/hp-arkui-use-id-in-get-resource-sync-api` (suggestion)
- file: `test/src/main/ets/app/app.ets`:396:7
- patch: shape=`add`, changed_lines=1

### Before (local context)
```
   390    async testOAuth2(prompt: PromptForUserAuthorize): Promise<void> {
   391      let redirectUri = "";
   392      let scopes: string[];
   393      let authEndPoint: string;
   394      let tokenEndPoint: string;
   395      if (this.context) {
   396        redirectUri = this.context.resourceManager.getStringSync($r('app.string.oauth2outlook_redirectUri'));
   397        // scopes = [
   398        //   this.context.resourceManager.getStringSync($r('app.string.oauth2outlook_scopes_imap')),
   399        //   this.context.resourceManager.getStringSync($r('app.string.oauth2outlook_scopes_smtp')),
   400        //   this.context.resourceManager.getStringSync($r('app.string.oauth2outlook_scopes_access')),
   401        //   this.context.resourceManager.getStringSync($r('app.string.oauth2outlook_scopes_openid')),
   402        // ]
```

### After (local context)
```
   390  
   391    async testOAuth2(prompt: PromptForUserAuthorize): Promise<void> {
   392      let redirectUri = "";
   393      let scopes: string[];
   394      let authEndPoint: string;
   395      let tokenEndPoint: string;
   396      if (this.context) {
   397        redirectUri = this.context.resourceManager.getStringSync($r('app.string.oauth2outlook_redirectUri'));
   398        // scopes = [
   399        //   this.context.resourceManager.getStringSync($r('app.string.oauth2outlook_scopes_imap')),
   400        //   this.context.resourceManager.getStringSync($r('app.string.oauth2outlook_scopes_smtp')),
   401        //   this.context.resourceManager.getStringSync($r('app.string.oauth2outlook_scopes_access')),
   402        //   this.context.resourceManager.getStringSync($r('app.string.oauth2outlook_scopes_openid')),
```

### Local diff
```diff
--- before/test/src/main/ets/app/app.ets
+++ after/test/src/main/ets/app/app.ets
@@ -1,3 +1,4 @@
+    for (const mail of mails) {
       const priority = await mail.getPriority();
       const messageId = await mail.getHeader("message-ID");
       logger.info("priority", priority, messageId);
@@ -22,4 +23,3 @@
       tokenEndPoint =
         this.context.resourceManager.getStringSync($r('app.string.oauth2outlook_TokenEndpoint'), `tenant`);
     } else {
-      redirectUri = 'https://login.microsoftonline.com/common/oauth2/nativeclient'
```

## faf3e10849d6
- project: `wifi_testapp`
- round_fixed: `3` (late)
- rule: `performance/hp-arkui-use-local-var-to-replace-state-var` (warn)
- file: `entry/src/main/ets/pages/subAppTest/rcpHttpTest.ets`:438:7
- patch: shape=`delete`, changed_lines=2

### Before (local context)
```
   432        resData += `错误码: ${err.code}\n`;
   433        resData += `错误信息: ${err.message}\n`;
   434        resData += `详细信息: ${JSON.stringify(err)}`;
   435        this.resData = resData;
   436        let codeValue = this.codeValue;
   437        codeValue = 'Error';
   438        this.codeValue = codeValue;
   439      }
   440    }
   441  
   442    clearResponse() {
   443      this.resData = '';
   444      this.codeValue = '';
```

### After (local context)
```
   432        resData += `详细信息: ${JSON.stringify(err)}`;
   433        this.resData = resData;
   434        let codeValue = this.codeValue;
   435        codeValue = 'Error';
   436        this.codeValue = codeValue;
   437      }
   438    }
   439  
   440    clearResponse() {
   441      this.resData = '';
   442      this.codeValue = '';
   443    }
   444  
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/subAppTest/rcpHttpTest.ets
+++ after/entry/src/main/ets/pages/subAppTest/rcpHttpTest.ets
@@ -1,5 +1,3 @@
-      isLoading = false;
-      this.isLoading = isLoading;
       const err = error as BusinessError;
       console.error(TAG, `RCP Request failed: ${JSON.stringify(err)}`);
       let resData = this.resData;
@@ -23,3 +21,5 @@
     // 取消网络监听
     if (this.netConnection) {
       try {
+        this.netConnection.unregister((error: Error) => {
+          if (error) {
```

## d577132945b3
- project: `wifi_testapp`
- round_fixed: `4` (late)
- rule: `performance/avoid-overusing-custom-component-check` (warn)
- file: `entry/src/main/ets/Component/headComponent.ets`:27:23
- patch: shape=`add`, changed_lines=1

### Before (local context)
```
    21  const TAG = "wifiTestApp [headComponent]"
    22  /**
    23   * head custom component Of WiFi test
    24   */
    25  
    26  @Component
    27  export default struct HeadComponent {
    28    private isActive: boolean = true;
    29    private icBackIsVisibility: boolean = true;
    30    private headName: string | Resource = '';
    31    @State isTouch: boolean = false;
    32  
    33    build() {
```

### After (local context)
```
    21  
    22  const TAG = "wifiTestApp [headComponent]"
    23  /**
    24   * head custom component Of WiFi test
    25   */
    26  
    27  @Component
    28  export default struct HeadComponent {
    29    private isActive: boolean = true;
    30    private icBackIsVisibility: boolean = true;
    31    private headName: string | Resource = '';
    32    @State isTouch: boolean = false;
    33  
```

### Local diff
```diff
--- before/entry/src/main/ets/Component/headComponent.ets
+++ after/entry/src/main/ets/Component/headComponent.ets
@@ -1,3 +1,4 @@
+ * See the License for the specific language governing permissions and
  * limitations under the License.
  */
 
@@ -22,4 +23,3 @@
         Image($r('app.media.ic_back'))
           .width($r('app.float.wh_value_30'))
           .height($r('app.float.wh_value_30'))
-      }
```

## b2aa56dc512a
- project: `wifi_testapp`
- round_fixed: `5` (late)
- rule: `performance/avoid-overusing-custom-component-check` (warn)
- file: `entry/src/main/ets/Component/infoView.ets`:32:15
- patch: shape=`add`, changed_lines=2

### Before (local context)
```
    26  interface IpInfo {
    27    key: Resource,
    28    value: string
    29  }
    30  
    31  @Component
    32  export struct InfoView {
    33    private infoList: IpInfo[] = []
    34  
    35    build() {
    36      Column() {
    37        ForEach(this.infoList , (item: IpInfo , index) => {
    38          Column() {
```

### After (local context)
```
    26  import ConfigData from '../Utils/ConfigData'
    27  
    28  interface IpInfo {
    29    key: Resource,
    30    value: string
    31  }
    32  
    33  @Component
    34  export struct InfoView {
    35    private infoList: IpInfo[] = []
    36  
    37    build() {
    38      Column() {
```

### Local diff
```diff
--- before/entry/src/main/ets/Component/infoView.ets
+++ after/entry/src/main/ets/Component/infoView.ets
@@ -1,3 +1,5 @@
+ */
+
 /**
  * info view Of WiFi test
  */
@@ -21,5 +23,3 @@
             .fontColor(Color.Black)
             .fontSize(20)
             .width(ConfigData.WH_100_100)
-          Text(item.value)
-            .fontColor(Color.Black)
```

## 86ac2b3f1437
- project: `Image`
- round_fixed: `3` (late)
- rule: `security/no-commented-code` (warn)
- file: `entry/src/main/ets/pages/ImageInterface/slice.ets`:15:1
- patch: shape=`add`, changed_lines=1

### Before (local context)
```
     9   * Unless required by applicable law or agreed to in writing, software
    10   * distributed under the License is distributed on an "AS IS" BASIS,
    11   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    12   * See the License for the specific language governing permissions and
    13   * limitations under the License.
    14   */
    15  /*
    16  import { FrameNode, NodeController, typeNode, UIContext } from '@kit.ArkUI';
    17  
    18  class MyImageNodeController extends NodeController {
    19    public uiContext: UIContext | null = null;
    20    public rootNode: FrameNode | null = null;
    21  
```

### After (local context)
```
     9   *
    10   * Unless required by applicable law or agreed to in writing, software
    11   * distributed under the License is distributed on an "AS IS" BASIS,
    12   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    13   * See the License for the specific language governing permissions and
    14   * limitations under the License.
    15   */
    16  /*
    17  import { FrameNode, NodeController, typeNode, UIContext } from '@kit.ArkUI';
    18  
    19  class MyImageNodeController extends NodeController {
    20    public uiContext: UIContext | null = null;
    21    public rootNode: FrameNode | null = null;
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/ImageInterface/slice.ets
+++ after/entry/src/main/ets/pages/ImageInterface/slice.ets
@@ -1,3 +1,4 @@
+ * Copyright (c) 2025 Huawei Device Co., Ltd.
  * Licensed under the Apache License, Version 2.0 (the "License");
  * you may not use this file except in compliance with the License.
  * You may obtain a copy of the License at
@@ -22,4 +23,3 @@
     this.rootNode = new FrameNode(uiContext);
     let node = typeNode.createNode(uiContext, 'Image');
     node.initialize($r('app.media.200both6'))
-      .width(100)
```

## b1052c3faa42
- project: `Image`
- round_fixed: `4` (late)
- rule: `security/no-commented-code` (warn)
- file: `entry/src/main/ets/pages/ImageInterface/slice.ets`:16:1
- patch: shape=`modify`, changed_lines=1

### Before (local context)
```
    10   * Unless required by applicable law or agreed to in writing, software
    11   * distributed under the License is distributed on an "AS IS" BASIS,
    12   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    13   * See the License for the specific language governing permissions and
    14   * limitations under the License.
    15   */
    16  /*
    17  import { FrameNode, NodeController, typeNode, UIContext } from '@kit.ArkUI';
    18  
    19  class MyImageNodeController extends NodeController {
    20    public uiContext: UIContext | null = null;
    21    public rootNode: FrameNode | null = null;
    22  
```

### After (local context)
```
    10   * Unless required by applicable law or agreed to in writing, software
    11   * distributed under the License is distributed on an "AS IS" BASIS,
    12   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    13   * See the License for the specific language governing permissions and
    14   * limitations under the License.
    15   */
    16   /*
    17  import { FrameNode, NodeController, typeNode, UIContext } from '@kit.ArkUI';
    18  
    19  class MyImageNodeController extends NodeController {
    20    public uiContext: UIContext | null = null;
    21    public rootNode: FrameNode | null = null;
    22  
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/ImageInterface/slice.ets
+++ after/entry/src/main/ets/pages/ImageInterface/slice.ets
@@ -10,7 +10,7 @@
  * See the License for the specific language governing permissions and
  * limitations under the License.
  */
-/*
+ /*
 import { FrameNode, NodeController, typeNode, UIContext } from '@kit.ArkUI';
 
 class MyImageNodeController extends NodeController {
```

## c37d3c0e97fd
- project: `YFree_HarmonyOS`
- round_fixed: `3` (late)
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

## 6c23aa2226d2
- project: `flutter_embedding`
- round_fixed: `3` (late)
- rule: `security/no-cycle` (warn)
- file: `flutter/src/main/ets/embedding/engine/FlutterEngine.ets`:24:1
- patch: shape=`modify`, changed_lines=2

### Before (local context)
```
    18  import common from '@ohos.app.ability.common';
    19  import resourceManager from '@ohos.resourceManager';
    20  import FlutterNapi from './FlutterNapi';
    21  import NavigationChannel from './systemchannels/NavigationChannel';
    22  import Log from '../../util/Log';
    23  import TestChannel from './systemchannels/TestChannel'
    24  import FlutterEngineConnectionRegistry from './FlutterEngineConnectionRegistry';
    25  import PluginRegistry from './plugins/PluginRegistry';
    26  import AbilityControlSurface from './plugins/ability/AbilityControlSurface';
    27  import TextInputChannel from './systemchannels/TextInputChannel';
    28  import TextInputPlugin from '../../plugin/editing/TextInputPlugin';
    29  import PlatformChannel from './systemchannels/PlatformChannel';
    30  import SystemChannel from './systemchannels/SystemChannel';
```

### After (local context)
```
    18  import FlutterLoader from './loader/FlutterLoader';
    19  import common from '@ohos.app.ability.common';
    20  import resourceManager from '@ohos.resourceManager';
    21  import FlutterNapi from './FlutterNapi';
    22  import NavigationChannel from './systemchannels/NavigationChannel';
    23  import Log from '../../util/Log';
    24  import TestChannel from './systemchannels/TestChannel'
    25  import FlutterEngineConnectionRegistry from './FlutterEngineConnectionRegistry2';
    26  import PluginRegistry from './plugins/PluginRegistry';
    27  import AbilityControlSurface from './plugins/ability/AbilityControlSurface';
    28  import TextInputChannel from './systemchannels/TextInputChannel';
    29  import TextInputPlugin from '../../plugin/editing/TextInputPlugin';
    30  import PlatformChannel from './systemchannels/PlatformChannel';
```

### Local diff
```diff
--- before/flutter/src/main/ets/embedding/engine/FlutterEngine.ets
+++ after/flutter/src/main/ets/embedding/engine/FlutterEngine.ets
@@ -1,3 +1,4 @@
+*/
 
 import LifecycleChannel from './systemchannels/LifecycleChannel2';
 import DartExecutor, { DartEntrypoint } from './dart/DartExecutor2';
@@ -10,7 +11,7 @@
 import NavigationChannel from './systemchannels/NavigationChannel';
 import Log from '../../util/Log';
 import TestChannel from './systemchannels/TestChannel'
-import FlutterEngineConnectionRegistry from './FlutterEngineConnectionRegistry';
+import FlutterEngineConnectionRegistry from './FlutterEngineConnectionRegistry2';
 import PluginRegistry from './plugins/PluginRegistry';
 import AbilityControlSurface from './plugins/ability/AbilityControlSurface';
 import TextInputChannel from './systemchannels/TextInputChannel';
@@ -22,4 +23,3 @@
 import LocalizationChannel from './systemchannels/LocalizationChannel';
 import AccessibilityChannel from './systemchannels/AccessibilityChannel';
 import LocalizationPlugin from '../../plugin/localization/LocalizationPlugin'
-import SettingsChannel from './systemchannels/SettingsChannel';
```

## 7e444ff3a723
- project: `ohos_dfu_library`
- round_fixed: `3` (late)
- rule: `security/no-cycle` (warn)
- file: `DfuLibrary/src/main/ets/dfu/ButtonlessDfuWithBondSharingImpl.ts`:19:1
- patch: shape=`modify`, changed_lines=2

### Before (local context)
```
    13   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    14   * See the License for the specific language governing permissions and
    15   * limitations under the License.
    16   */
    17  
    18  import { ButtonlessDfuImpl } from './ButtonlessDfuImplBase'
    19  import { DfuBaseService } from './DfuServiceProvider'
    20  import { SecureDfuImpl } from './SecureDfuImpl'
    21  import ble from '@ohos.bluetooth.ble';
    22  import hilog from '@ohos.hilog';
    23  
    24  export class ButtonlessDfuWithBondSharingImpl extends ButtonlessDfuImpl {
    25    /**
```

### After (local context)
```
    13   * distributed under the License is distributed on an "AS IS" BASIS,
    14   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    15   * See the License for the specific language governing permissions and
    16   * limitations under the License.
    17   */
    18  
    19  import { ButtonlessDfuImpl } from './ButtonlessDfuImplBase'
    20  import { DfuBaseService } from './DfuBaseService'
    21  import { SecureDfuImpl } from './SecureDfuImpl'
    22  import ble from '@ohos.bluetooth.ble';
    23  import hilog from '@ohos.hilog';
    24  
    25  export class ButtonlessDfuWithBondSharingImpl extends ButtonlessDfuImpl {
```

### Local diff
```diff
--- before/DfuLibrary/src/main/ets/dfu/ButtonlessDfuWithBondSharingImpl.ts
+++ after/DfuLibrary/src/main/ets/dfu/ButtonlessDfuWithBondSharingImpl.ts
@@ -1,3 +1,4 @@
+ * you may not use this file except in compliance with the License.
  * You may obtain a copy of the License at
  *
  *     http://www.apache.org/licenses/LICENSE-2.0
@@ -10,7 +11,7 @@
  */
 
 import { ButtonlessDfuImpl } from './ButtonlessDfuImplBase'
-import { DfuBaseService } from './DfuServiceProvider'
+import { DfuBaseService } from './DfuBaseService'
 import { SecureDfuImpl } from './SecureDfuImpl'
 import ble from '@ohos.bluetooth.ble';
 import hilog from '@ohos.hilog';
@@ -22,4 +23,3 @@
   static DEFAULT_BUTTONLESS_DFU_SERVICE_UUID: string = SecureDfuImpl.DEFAULT_DFU_SERVICE_UUID;
   /**
    * The UUID of the Secure Buttonless DFU characteristic with bond sharing from SDK 14 or newer.
-   */
```
