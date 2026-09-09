# @performance/hp-arkui-no-state-var-access-in-loop

Static repair references: 4. These are examples, not patches to copy.
Inspect repository context and checker evidence before adapting an example.

## Example 1: `pair_1fbfecf877ddecea`

避免在for、while等循环逻辑中频繁读取状态变量。

### Triggering pattern

```arkts
for ( let indexNumber = 1 ; indexNumber < 101 ; indexNumber ++ ) {
            let message: string = apiItem.api!();
            apiItem.result = message;
            prompt.showToast({ message : message });
            this.currentIndex += 1;
            this.changeIndex = this.currentIndex;
            AppStorage.setOrCreate('stressNumber' , indexNumber);
            console.log(TAG , apiItem.method + "indexNumber is ------: " + indexNumber);
          }
```

### Repair pattern

```arkts
let currentIndex = this.currentIndex
          for ( let indexNumber = 1 ; indexNumber < 101 ; indexNumber ++ ) {
            let message: string = apiItem.api!();
            apiItem.result = message;
            prompt.showToast({ message : message });
            currentIndex += 1;
            this.changeIndex = currentIndex;
            AppStorage.setOrCreate('stressNumber' , indexNumber);
            console.log(TAG , apiItem.method + "indexNumber is ------: " + indexNumber);
          }
          this.currentIndex = currentIndex
```

### Rationale

循环中反复读取状态变量this.currentIndex， 应该在循环外读取该值，然后替代this.currentIndex

## Example 2: `pair_7b9bfc0fc7ad7883`

避免在for、while等循环逻辑中频繁读取状态变量。

### Triggering pattern

```arkts
for ( this.openHotspotNumber ; this.openHotspotNumber < this.testNumbers ; this.openHotspotNumber ++ ) {
      if ( !this.hotspotLoopState ) {
        console.log(TAG , "测试结束------------")
        break;
      } else {
        let wifiState = wifiManager.isWifiActive()
        if ( wifiState ) {
          wifi.disableWifi()
          console.log(TAG , "wifi当前已使能，已经去使能，正常开始测试------")
        } else {
          console.log(TAG , "wifi当前未使能，正常开始测试------")
        }
        await sleep(3)

        funcMessage = wifi.enableHotspot()
        // 打时间戳
        this.open_StartTime = new Date().getTime()
        console.log(TAG , "第" + (this.openHotspotNumber + 1) + "次热点使能-----")
        console.log(TAG , "第" + (this.openHotspotNumber + 1) + "次热点使能开始时间: " + this.open_StartTime + "ms")
        this.hotspotMessageLog += "第" + (this.openHotspotNumber + 1) + "次热点使能结果：" + funcMessage + "\n"
        console.log(TAG , "第" + (this.openHotspotNumber + 1) + "次热点使能结果：" + funcMessage)
        await sleep(10)
        this.hotspotMessage = AppStorage.get("hotspotMessage") ! //非空断言操作符
        // prompt.showToast( { message : funcMessage } )
        if ( this.hotspotMessage == "active" ) {
          this.open_SpendTime = this.open_EndTime - this.open_StartTime
          this.hotspotMessageLog += "第" + (this.openHotspotNumber + 1) + "次热点使能耗时: " + this.open_SpendTime + "ms" + "\n"
          console.log(TAG , "第" + (this.openHotspotNumber + 1) + "次热点使能耗时: " + this.open_SpendTime + "ms")
          this.openSuccessNumber = this.openSuccessNumber + 1
          this.hotspotMessageLog += "热点使能成功的次数：" + this.openSuccessNumber + "\n"
          console.log(TAG , "热点使能成功的次数：" + this.openSuccessNumber)
          funcMessage = wifi.disableHotspot()
          this.closeHotspotNumber = this.closeHotspotNumber + 1
          this.close_StartTime = new Date().getTime()
          console.log(TAG , "第" + this.closeHotspotNumber + "次热点去使能-----")
          console.log(TAG , "第" + this.closeHotspotNumber + "次热点去使能开始时间: " + this.close_StartTime + "ms")
          this.hotspotMessageLog += "第" + this.closeHotspotNumber + "次热点去使能结果：" + funcMessage + "\n"
          console.log(TAG , "第" + this.closeHotspotNumber + "次热点去使能结果：" + funcMessage)
          console.log(TAG , "closeHotspotNumber: " + this.closeHotspotNumber)
          await sleep(10)
          this.hotspotMessage = AppStorage.get("hotspotMessage") ! //非空断言操作符
          if ( this.hotspotMessage == "inactive" ) {
            this.close_SpendTime = this.close_EndTime - this.close_StartTime
            this.hotspotMessageLog += "第" + this.closeHotspotNumber + "次热点去使能耗时: " + this.close_SpendTime + "ms" + "\n"
            console.log(TAG , "第" + this.closeHotspotNumber + "次热点去使能耗时: " + this.close_SpendTime + "ms")
            this.closeSuccessNumber = this.closeSuccessNumber + 1
            this.hotspotMessageLog += "热点去使能成功的次数：" + this.closeSuccessNumber + "\n"
            console.log(TAG , "热点去使能成功的次数：" + this.closeSuccessNumber)
            await sleep(7)
          } else {
            this.closeFailNumber = this.closeFailNumber + 1
            console.log(TAG , "热点去使能失败的次数：" + this.closeFailNumber)
            console.log(TAG , "第" + this.closeHotspotNumber + "次热点去使能失败")
          }
        } else if ( this.hotspotMessage == "inactive" ) {
          this.openFailNumber = this.openFailNumber + 1
          console.log(TAG , "热点使能失败的次数：" + this.openFailNumber)
          console.log(TAG , "第" + (this.openHotspotNumber + 1) + "次热点使能失败")
        } else {
          console.log("第" + (this.openHotspotNumber + 1) + "次开热点后状态不清楚");
        }
      }
    }
```

### Repair pattern

```arkts
let hotspotLoopState = this.hotspotLoopState
    let openHotspotNumber = this.openHotspotNumber
    let open_StartTime = this.open_StartTime
    let open_EndTime = this.open_EndTime
    let open_SpendTime = this.open_SpendTime
    let openSuccessNumber = this.openSuccessNumber
    let closeHotspotNumber = this.closeHotspotNumber
    let close_StartTime = this.close_StartTime
    let close_SpendTime = this.close_SpendTime
    let close_EndTime = this.close_EndTime
    let hotspotMessage = this.hotspotMessage
    let closeSuccessNumber = this.closeSuccessNumber
    let closeFailNumber = this.closeFailNumber
    let openFailNumber = this.openFailNumber
    for ( this.openHotspotNumber ; this.openHotspotNumber < this.testNumbers ; this.openHotspotNumber ++ ) {
      if ( !hotspotLoopState ) {
        console.log(TAG , "测试结束------------")
        break;
      } else {
        let wifiState = wifiManager.isWifiActive()
        if ( wifiState ) {
          wifi.disableWifi()
          console.log(TAG , "wifi当前已使能，已经去使能，正常开始测试------")
        } else {
          console.log(TAG , "wifi当前未使能，正常开始测试------")
        }
        await sleep(3)

        funcMessage = wifi.enableHotspot()
        // 打时间戳
        this.open_StartTime = new Date().getTime()
        console.log(TAG , "第" + (openHotspotNumber + 1) + "次热点使能-----")
        console.log(TAG , "第" + (openHotspotNumber + 1) + "次热点使能开始时间: " + open_StartTime + "ms")
        this.hotspotMessageLog += "第" + (openHotspotNumber + 1) + "次热点使能结果：" + funcMessage + "\n"
        console.log(TAG , "第" + (openHotspotNumber + 1) + "次热点使能结果：" + funcMessage)
        await sleep(10)
        this.hotspotMessage = AppStorage.get("hotspotMessage") ! //非空断言操作符
        // prompt.showToast( { message : funcMessage } )
        if ( hotspotMessage == "active" ) {
          open_SpendTime = open_EndTime - open_StartTime
          this.hotspotMessageLog += "第" + (openHotspotNumber + 1) + "次热点使能耗时: " + open_SpendTime + "ms" + "\n"
          console.log(TAG , "第" + (openHotspotNumber + 1) + "次热点使能耗时: " + open_SpendTime + "ms")
          openSuccessNumber = openSuccessNumber + 1
          this.hotspotMessageLog += "热点使能成功的次数：" + openSuccessNumber + "\n"
          console.log(TAG , "热点使能成功的次数：" + openSuccessNumber)
          funcMessage = wifi.disableHotspot()
          closeHotspotNumber = closeHotspotNumber + 1
          close_StartTime = new Date().getTime()
          console.log(TAG , "第" + closeHotspotNumber + "次热点去使能-----")
          console.log(TAG , "第" + closeHotspotNumber + "次热点去使能开始时间: " + close_StartTime + "ms")
          this.hotspotMessageLog += "第" + closeHotspotNumber + "次热点去使能结果：" + funcMessage + "\n"
          console.log(TAG , "第" + closeHotspotNumber + "次热点去使能结果：" + funcMessage)
          console.log(TAG , "closeHotspotNumber: " + closeHotspotNumber)
          await sleep(10)
          hotspotMessage = AppStorage.get("hotspotMessage") ! //非空断言操作符
          if ( hotspotMessage == "inactive" ) {
            close_SpendTime = close_EndTime - close_StartTime
            this.hotspotMessageLog += "第" + closeHotspotNumber + "次热点去使能耗时: " + close_SpendTime + "ms" + "\n"
            console.log(TAG , "第" + closeHotspotNumber + "次热点去使能耗时: " + close_SpendTime + "ms")
            closeSuccessNumber = closeSuccessNumber + 1
            this.hotspotMessageLog += "热点去使能成功的次数：" + closeSuccessNumber + "\n"
            console.log(TAG , "热点去使能成功的次数：" + closeSuccessNumber)
            await sleep(7)
          } else {
            closeFailNumber = closeFailNumber + 1
            console.log(TAG , "热点去使能失败的次数：" + closeFailNumber)
            console.log(TAG , "第" + closeHotspotNumber + "次热点去使能失败")
          }
        } else if ( this.hotspotMessage == "inactive" ) {
          openFailNumber = openFailNumber + 1
          console.log(TAG , "热点使能失败的次数：" + openFailNumber)
          console.log(TAG , "第" + (openHotspotNumber + 1) + "次热点使能失败")
        } else {
          console.log("第" + (openHotspotNumber + 1) + "次开热点后状态不清楚");
        }
      }
    }

    this.hotspotLoopState = hotspotLoopState
    this.openHotspotNumber = openHotspotNumber
    this.open_StartTime = open_StartTime
    this.open_EndTime = open_EndTime
    this.open_SpendTime = open_SpendTime
    this.openSuccessNumber = openSuccessNumber
    this.closeHotspotNumber = closeHotspotNumber
    this.close_StartTime = close_StartTime
    this.close_SpendTime = close_SpendTime
    this.close_EndTime = close_EndTime
    this.hotspotMessage = hotspotMessage
    this.closeSuccessNumber = closeSuccessNumber
    this.closeFailNumber = closeFailNumber
    this.openFailNumber = openFailNumber
```

### Rationale

循环中反复读取状态变量， 应该在循环外读取该值，然后替代

## Example 3: `pair_7f25871a725ad300`

避免在for、while等循环逻辑中频繁读取状态变量。

### Triggering pattern

```arkts
import hilog from '@ohos.hilog'
@Entry
@Component
struct MyComponent{
  @State message: string = '';
  build() {
    Column() {
      Button('点击打印日志')
        .onClick(() => {
          this.message = 'click';
          for (let i = 0; i < 10; i++) {
            hilog.info(0x0000, 'TAG', '%{public}s', this.message);
          }
        })
        .width('90%')
        .backgroundColor(Color.Blue)
        .fontColor(Color.White)
        .margin({
          top: 10
        })
    }
    .justifyContent(FlexAlign.Start)
    .alignItems(HorizontalAlign.Center)
    .margin({
      top: 15
    })
  }
}
```

### Repair pattern

```arkts
import hilog from '@ohos.hilog'

@Entry
@Component
struct MyComponent{
  @State message: string = '';
  build() {
    Column() {
      Button('点击打印日志')
        .onClick(() => {
          this.message = 'click';
          let logMessage: string = this.message;
          for (let i = 0; i < 10; i++) {
            hilog.info(0x0000, 'TAG', '%{public}s', logMessage);
          }
        })
        .width('90%')
        .backgroundColor(Color.Blue)
        .fontColor(Color.White)
        .margin({
          top: 10
        })
    }
    .justifyContent(FlexAlign.Start)
    .alignItems(HorizontalAlign.Center)
    .margin({
      top: 15
    })
  }
}
```

### Rationale

循环中反复读取状态变量this.message， 应该在循环外读取该值，然后替代this.message

## Example 4: `pair_7f7e59c8ee08c328`

避免在for、while等循环逻辑中频繁读取状态变量。

### Triggering pattern

```arkts
for ( this.openHotspotNumber ; this.openHotspotNumber < this.testNumbers ; this.openHotspotNumber ++ ) {
  if (!this.hotspotLoopState) {
    console.log(TAG, "测试结束------------")
    break;
  } else {
  }
}
```

### Repair pattern

```arkts
let state = this.hotspotLoopState
for ( this.openHotspotNumber ; this.openHotspotNumber < this.testNumbers ; this.openHotspotNumber ++ ) {
  if ( !state ) {
    console.log(TAG , "测试结束------------")
    break;
  } else {
  }
}
```

### Rationale

循环中反复读取状态变量this.hotspotLoopState， 应该在循环外读取该值，然后替代this.hotspotLoopState
