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