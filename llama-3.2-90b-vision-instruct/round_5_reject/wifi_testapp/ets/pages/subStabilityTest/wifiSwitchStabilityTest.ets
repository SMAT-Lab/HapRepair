

// ...

@Entry
@Component
struct WiFiSwitchStabilityTest {
  private testItem: TestData = (router.getParams() as myParams).testItem
  open_StartTime: number = 0
  open_EndTime: number = 0
  open_SpendTime: number = 0
  close_StartTime: number = 0
  close_EndTime: number = 0
  close_SpendTime: number = 0
  openWifiNumber: number = 0 // 打开WiFi的次数
  closeWifiNumber: number = 0 // 关闭WiFi的次数
  openSuccessNumber: number = 0 // 打开WiFi的成功次数
  openFailNumber: number = 0 // 打开WiFi的失败次数
  closeSuccessNumber: number = 0 // 关闭WiFi的成功次数
  closeFailNumber: number = 0 // 关闭WiFi的失败次数
  message: string = "测试结果:"
  testNumbers: number = 30 //测试次数
  wifiState: boolean = wifiManager.isWifiActive()
  testNumbersValue: number = 30
  file!: number
  filePathDir!: string
  private file!: fs.File
  @StorageLink("pathDir") pathDir: string = ""
  stateMessage: string = ""
  stateMessageLog: string = ""
  switchLoopState: Boolean = true

  // ...

  async forLoop() {
    let openWifiNumber = this.openWifiNumber
    let closeWifiNumber = this.closeWifiNumber
    let openSuccessNumber = this.openSuccessNumber
    let openFailNumber = this.openFailNumber
    let closeSuccessNumber = this.closeSuccessNumber
    let closeFailNumber = this.closeFailNumber
    let open_StartTime = this.open_StartTime
    let open_EndTime = this.open_EndTime
    let close_StartTime = this.close_StartTime
    let close_EndTime = this.close_EndTime
    let stateMessageLog: string = ""
    let stateMessage = ""
    let funcMessage = false // 函数信息
    let testNumbersValue: number = 30 //测试次数
    for ( openWifiNumber ; openWifiNumber < testNumbersValue ; openWifiNumber ++ ) {
      if ( !switchLoopState ) {
        openWifiNumber = 0; 
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
        funcMessage = wifi.enableWifi()
        // 打时间戳
        open_StartTime = new Date().getTime()
        console.log(TAG , "第" + (openWifiNumber + 1) + "次WIFI使能-----")
        console.log(TAG , "第" + (openWifiNumber + 1) + "次WIFI使能开始时间: " + open_StartTime + "ms")
        stateMessageLog += "第" + (openWifiNumber + 1) + "次WIFI使能结果：" + funcMessage + "\n"
        console.log(TAG , "第" + (openWifiNumber + 1) + "次WIFI使能结果：" + funcMessage)
        await sleep(15)
        stateMessage = AppStorage.get("stateMessage")! //非空断言操作符
        if ( stateMessage == "active" ) {
          let open_SpendTime = open_EndTime - open_StartTime
          stateMessageLog += "第" + (openWifiNumber + 1) + "次WIFI使能耗时: " + open_SpendTime + "ms" + "\n"
          console.log(TAG , "第" + (openWifiNumber + 1) + "次WIFI使能耗时: " + open_SpendTime + "ms")
          openSuccessNumber = openSuccessNumber + 1
          stateMessageLog += "使能成功的次数：" + openSuccessNumber + "\n"
          console.log(TAG , "使能成功的次数：" + openSuccessNumber)
          funcMessage = wifi.disableWifi()
          closeWifiNumber = closeWifiNumber + 1
          close_StartTime = new Date().getTime()
          console.log(TAG , "第" + closeWifiNumber + "次WIFI去使能-----")
          console.log(TAG , "第" + closeWifiNumber + "次WIFI去使能开始时间: " + close_StartTime + "ms")
          stateMessageLog += "第" + closeWifiNumber + "次数热点去使能接口的调用结果：" + funcMessage + "\n"
          console.log(TAG , "第" + closeWifiNumber + "次数热点去使能接口的调用结果：" + funcMessage)
          console.log(TAG , "closeWifiNumber: " + closeWifiNumber)
          await sleep(15)
          stateMessage = AppStorage.get("stateMessage")! //非空断言操作符
          if ( stateMessage == "inactive" ) {
            let close_SpendTime = close_EndTime - close_StartTime
            stateMessageLog += "第" + closeWifiNumber + "次WIFI去使能耗时: " + close_SpendTime + "ms" + "\n"
            console.log(TAG , "第" + closeWifiNumber + "次WIFI去使能耗时: " + close_SpendTime + "ms")
            closeSuccessNumber = closeSuccessNumber + 1
            stateMessageLog += "去使能成功的次数：" + closeSuccessNumber + "\n"
            console.log(TAG , "去使能成功的次数：" + closeSuccessNumber)
            await sleep(7)
          } else {
            closeFailNumber = closeFailNumber + 1
            console.log(TAG , "去使能失败的次数：" + closeFailNumber)
            console.log(TAG , "第" + closeWifiNumber + "次WIFI去使能失败")
          }
        } else if ( stateMessage == "inactive" ) {
          openFailNumber = openFailNumber + 1
          console.log(TAG , "使能失败的次数：" + openFailNumber)
          console.log(TAG , "第" + (openWifiNumber + 1) + "次WIFI使能失败")
        } else {
          console.log("第" + (openWifiNumber + 1) + "次开wifi后状态不清楚");
        }
      }
    }
    console.log(TAG , "测试结束 -------------")
    this.stopContinuousTask();
    AppStorage.setOrCreate("stateMessageLog" , stateMessageLog)
    AppStorage.setOrCreate("openWifiNumber" , openWifiNumber)
    AppStorage.setOrCreate("closeWifiNumber" , closeWifiNumber)
    AppStorage.setOrCreate("openSuccessNumber" , openSuccessNumber)
    AppStorage.setOrCreate("openFailNumber" , openFailNumber)
    AppStorage.setOrCreate("closeSuccessNumber" , closeSuccessNumber)
    AppStorage.setOrCreate("closeFailNumber" , closeFailNumber)
    AppStorage.setOrCreate("open_StartTime" , open_StartTime)
    AppStorage.setOrCreate("close_StartTime" , close_StartTime)
  }

  // ...
}