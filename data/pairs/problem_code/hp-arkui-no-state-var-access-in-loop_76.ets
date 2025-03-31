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