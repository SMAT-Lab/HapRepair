# Repair semantic-eval sample (gpt-5.1)

- n=5, seed=1, max_round=1

## 7e555c55ee36
- project: `Image`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-redundant-state-var` (suggestion)
- file: `entry/src/main/ets/pages/testcase/ImageTestCase007.ets`:23:3
- patch: shape=`delete`, changed_lines=1

### Before (local context)
```
    17  /**
    18   * 验证Image的基础功能
    19   */
    20  @Entry
    21  @Component
    22  struct ImageExample {
    23    @State imageSource: ResourceStr | undefined = undefined;
    24    @State imageFits: ImageFit[] = [];
    25    @State imageFitIndex: number = 0;
    26    @State imageRepeats: ImageRepeat[] = [];
    27    @State imageRepeatIndex: number = 0;
    28    @State imageRenderMode: ImageRenderMode[] = [];
    29    @State imageRenderModeIndex: number = 0;
```

### After (local context)
```
    17  /**
    18   * 验证Image的基础功能
    19   */
    20  @Entry
    21  @Component
    22  struct ImageExample {
    23    @State imageFits: ImageFit[] = [];
    24    @State imageFitIndex: number = 0;
    25    @State imageRepeats: ImageRepeat[] = [];
    26    @State imageRepeatIndex: number = 0;
    27    @State imageRenderMode: ImageRenderMode[] = [];
    28    @State imageRenderModeIndex: number = 0;
    29    @State imageInterpolations: ImageInterpolation[] = [];
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/testcase/ImageTestCase007.ets
+++ after/entry/src/main/ets/pages/testcase/ImageTestCase007.ets
@@ -10,7 +10,6 @@
 @Entry
 @Component
 struct ImageExample {
-  @State imageSource: ResourceStr | undefined = undefined;
   @State imageFits: ImageFit[] = [];
   @State imageFitIndex: number = 0;
   @State imageRepeats: ImageRepeat[] = [];
@@ -23,3 +22,4 @@
   fillColorIndex: number = 0;
 
   aboutToAppear(): void {
+    this.pushDataToImageFits();
```

## 8a8aac31553c
- project: `bluetoothtest`
- round_fixed: `1` (early)
- rule: `performance/hp-arkui-remove-unchanged-state-var` (suggestion)
- file: `entry/src/main/ets/pages/subBenchmarkTestManager/bleBenchmarkTestManager.ets`:38:3
- patch: shape=`delete`, changed_lines=1

### Before (local context)
```
    32   */
    33  
    34  @Entry
    35  @Component
    36  struct BleBenchmarkTestManager {
    37    private testItem: TestData = (router.getParams() as myParams).testItem
    38    @State changeIndex: number = - 1
    39    @StorageLink("bleBenchmarkTestMessage") bleBenchmarkTestMessage: string = ""
    40  
    41    build() {
    42      Column() {
    43        Stack({ alignContent : Alignment.TopStart }) {
    44          TestImageDisplay({ testItem : this.testItem })
```

### After (local context)
```
    32   */
    33  
    34  @Entry
    35  @Component
    36  struct BleBenchmarkTestManager {
    37    private testItem: TestData = (router.getParams() as myParams).testItem
    38    @StorageLink("bleBenchmarkTestMessage") bleBenchmarkTestMessage: string = ""
    39  
    40    build() {
    41      Column() {
    42        Stack({ alignContent : Alignment.TopStart }) {
    43          TestImageDisplay({ testItem : this.testItem })
    44          PageTitle({ testItem : this.testItem })
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/subBenchmarkTestManager/bleBenchmarkTestManager.ets
+++ after/entry/src/main/ets/pages/subBenchmarkTestManager/bleBenchmarkTestManager.ets
@@ -10,7 +10,6 @@
 @Component
 struct BleBenchmarkTestManager {
   private testItem: TestData = (router.getParams() as myParams).testItem
-  @State changeIndex: number = - 1
   @StorageLink("bleBenchmarkTestMessage") bleBenchmarkTestMessage: string = ""
 
   build() {
@@ -23,3 +22,4 @@
       Stack().height("1vp").backgroundColor("#000000");
 
       Column() {
+        Scroll() {
```

## 06c6a312d13f
- project: `audio_suite`
- round_fixed: `1` (early)
- rule: `security/no-unsafe-hash` (error)
- file: `entry/src/main/ets/pages/AutoTest.ets`:210:37
- patch: shape=`modify`, changed_lines=41

### Before (local context)
```
   204    let file = fs.openSync(filePath, fs.OpenMode.READ_WRITE);
   205    let buf = new ArrayBuffer(40960);
   206    fs.readSync(file.fd, buf);
   207    fs.closeSync(file);
   208  
   209    let mdAlgName = 'MD5'; // 摘要算法名。
   210    let md = cryptoFramework.createMd(mdAlgName);
   211    // 数据量较少时，可以只做一次update，将数据全部传入，接口未对入参长度做限制。
   212    md.updateSync({ data: new Uint8Array(buf) });
   213    let mdResult = md.digestSync();
   214    Logger.info(TAG, '[Sync]:Md result:' + mdResult.data);
   215    let md5Str = buffer.from(mdResult.data.buffer).toString('hex');
   216    Logger.info(TAG, '[Sync]:Md string result:' + md5Str);
```

### After (local context)
```
   204    let file = fs.openSync(filePath, fs.OpenMode.READ_WRITE);
   205    let buf = new ArrayBuffer(40960);
   206    fs.readSync(file.fd, buf);
   207    fs.closeSync(file);
   208  
   209    let mdAlgName = 'SHA256'; // 摘要算法名。
   210    let md = cryptoFramework.createMd(mdAlgName);
   211    // 数据量较少时，可以只做一次update，将数据全部传入，接口未对入参长度做限制。
   212    md.updateSync({ data: new Uint8Array(buf) });
   213    let mdResult = md.digestSync();
   214    Logger.info(TAG, '[Sync]:Md result:' + mdResult.data);
   215    let md5Str = buffer.from(mdResult.data.buffer).toString('hex');
   216    Logger.info(TAG, '[Sync]:Md string result:' + md5Str);
```

### Local diff
```diff
--- before/entry/src/main/ets/pages/AutoTest.ets
+++ after/entry/src/main/ets/pages/AutoTest.ets
@@ -9,7 +9,7 @@
   fs.readSync(file.fd, buf);
   fs.closeSync(file);
 
-  let mdAlgName = 'MD5'; // 摘要算法名。
+  let mdAlgName = 'SHA256'; // 摘要算法名。
   let md = cryptoFramework.createMd(mdAlgName);
   // 数据量较少时，可以只做一次update，将数据全部传入，接口未对入参长度做限制。
   md.updateSync({ data: new Uint8Array(buf) });
```

## 7abe50014406
- project: `flutter_embedding`
- round_fixed: `1` (early)
- rule: `security/no-cycle` (warn)
- file: `flutter/src/main/ets/embedding/engine/systemchannels/MouseCursorChannel.ets`:16:1
- patch: shape=`modify`, changed_lines=1

### Before (local context)
```
    10  
    11  import HashMap from '@ohos.util.HashMap';
    12  import MethodCall from '../../../plugin/common/MethodCall';
    13  import MethodChannel, { MethodCallHandler, MethodResult } from '../../../plugin/common/MethodChannel';
    14  import StandardMethodCodec from '../../../plugin/common/StandardMethodCodec';
    15  import Log from '../../../util/Log';
    16  import DartExecutor from '../dart/DartExecutor';
    17  
    18  const TAG: string = 'MouseCursorChannel';
    19  
    20  export default class MouseCursorChannel implements MethodCallHandler {
    21    public channel: MethodChannel;
    22    private mouseCursorMethodHandler: MouseCursorMethodHandler | null = null;
```

### After (local context)
```
    10  
    11  import HashMap from '@ohos.util.HashMap';
    12  import MethodCall from '../../../plugin/common/MethodCall';
    13  import MethodChannel, { MethodCallHandler, MethodResult } from '../../../plugin/common/MethodChannel';
    14  import StandardMethodCodec from '../../../plugin/common/StandardMethodCodec';
    15  import Log from '../../../util/Log';
    16  import DartExecutor from '../dart/DartExecutorC';
    17  
    18  const TAG: string = 'MouseCursorChannel';
    19  
    20  export default class MouseCursorChannel implements MethodCallHandler {
    21    public channel: MethodChannel;
    22    private mouseCursorMethodHandler: MouseCursorMethodHandler | null = null;
```

### Local diff
```diff
--- before/flutter/src/main/ets/embedding/engine/systemchannels/MouseCursorChannel.ets
+++ after/flutter/src/main/ets/embedding/engine/systemchannels/MouseCursorChannel.ets
@@ -10,7 +10,7 @@
 import MethodChannel, { MethodCallHandler, MethodResult } from '../../../plugin/common/MethodChannel';
 import StandardMethodCodec from '../../../plugin/common/StandardMethodCodec';
 import Log from '../../../util/Log';
-import DartExecutor from '../dart/DartExecutor';
+import DartExecutor from '../dart/DartExecutorC';
 
 const TAG: string = 'MouseCursorChannel';
 
```

## 471f3a1352f2
- project: `ohos_mail_base`
- round_fixed: `1` (early)
- rule: `security/no-commented-code` (warn)
- file: `lib/src/main/ets/format/rtf/font.ts`:170:5
- patch: shape=`modify`, changed_lines=2

### Before (local context)
```
   164      [65000, "utf-7"],
   165      [65001, "utf-8"]
   166    ]);
   167    private _encoding: Encoding;
   168  
   169    static get DefaultEncoding(): Encoding {
   170      // return Encoding.getEncoding("Windows-1252");
   171      return Encoding.getEncoding("utf-8");
   172    }
   173  
   174    get encoding() {
   175      return this._encoding;
   176    }
```

### After (local context)
```
   164      [57011, "x-iscii-pa"],
   165      [65000, "utf-7"],
   166      [65001, "utf-8"]
   167    ]);
   168    private _encoding: Encoding;
   169  
   170    static get DefaultEncoding(): Encoding {
   171      return Encoding.getEncoding("utf-8");
   172    }
   173  
   174    get encoding() {
   175      return this._encoding;
   176    }
```

### Local diff
```diff
--- before/lib/src/main/ets/format/rtf/font.ts
+++ after/lib/src/main/ets/format/rtf/font.ts
@@ -1,3 +1,4 @@
+    [57005, "x-iscii-te"],
     [57006, "x-iscii-as"],
     [57007, "x-iscii-or"],
     [57008, "x-iscii-ka"],
@@ -10,7 +11,6 @@
   private _encoding: Encoding;
 
   static get DefaultEncoding(): Encoding {
-    // return Encoding.getEncoding("Windows-1252");
     return Encoding.getEncoding("utf-8");
   }
 
```
