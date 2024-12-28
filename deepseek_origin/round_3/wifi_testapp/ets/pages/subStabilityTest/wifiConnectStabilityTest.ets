```arkts

/**
 * Copyright (c) 2022 Shenzhen Kaihong Digital Industry Development Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import { PageTitle } from '../../Component/pageTitle';
import { TestImageDisplay } from '../../Component/testImageDisplay';
import router from '@ohos.router';
import ConfigData from '../../Utils/ConfigData';
import { TestData } from '../../entryability/model/testData';
import wifi from '@ohos.wifi';
import wifiManager from '@ohos.wifiManager';
import prompt from '@system.prompt';
import common from '@ohos.app.ability.common';
import wantAgent , { WantAgent } from '@ohos.app.ability.wantAgent';
import backgroundTaskManager from '@ohos.resourceschedule.backgroundTaskManager';
import { BusinessError } from '@ohos.base';
import abilityAccessCtrl from '@ohos.abilityAccessCtrl';
import notificationManager from '@ohos.notificationManager';

interface myParams extends Object {
  testItem: TestData
}

/**
 * WiFiConnectStabilityTest Page Of wifi test
 */
const TAG = '[WiFiConnectStabilityTest]'

@Entry
@Component
struct WiFiConnectStabilityTest {
  private testItem: TestData = (router.getParams() as myParams).testItem
  @State connect_StartTime: number = 0
  @State connect_EndTime: number = 0
  @State connect_SpendTime: number = 0
  @State disconnect_StartTime: number = 0
  @State disconnect_EndTime: number = 0
  @State disconnect_SpendTime: number = 0
  @State connectNumber: number = 0 // 打开WiFi的次数
  @State disconnectNumber: number = 0 // 关闭WiFi的次数
  @State connectSuccessNumber: number = 0 // 打开WiFi的成功次数
  @State connectFailNumber: number = 0 // 打开WiFi的失败次数
  @State disconnectSuccessNumber: number = 0 // 关闭WiFi的成功次数
  @State disconnectFailNumber: number = 0 // 关闭WiFi的失败次数
  @State testNumbers: number = 30 //测试次数
  successTimes: number = 0
  failTimes: number = 0
  @State connectionMessage: string = ""
  @State connectionMessageLog: string = ""
  @State connectionLoopState: Boolean = true
  @State showList: boolean = false;
  @State ipInfo: number = 0
  @StorageLink('wifiConfig') wifiConfig: wifiManager.WifiDeviceConfig = {
    ssid : "TP-LINK_6365" ,
    preSharedKey : "12345678" ,
    securityType : 3 ,
  }

  aboutToAppear() {
    // 请求发送通知的许可
    notificationManager.requestEnableNotification().then(() => {
      console.info(`[EntryAbility] requestEnableNotification success`);
      // 申请WiFi相关权限
      let atManager = abilityAccessCtrl.createAtManager();
      try {
        atManager.requestPermissionsFromUser(getContext(this) , [
          'ohos.permission.INTERNET',
          'ohos.permission.SET_WIFI_INFO',
          'ohos.permission.GET_WIFI_INFO',
          'ohos.permission.MANAGE_WIFI_CONNECTION',
          'ohos.permission.GET_WIFI_PEERS_MAC',
          'ohos.permission.MANAGE_WIFI_CONNECTION',
          'ohos.permission.SET_WIFI_CONFIG',
          "ohos.permission.GET_WIFI_CONFIG",
          "ohos.permission.GET_WIFI_LOCAL_MAC",
          "ohos.permission.MANAGE_WIFI_HOTSPOT",
          "ohos.permission.GET_WIFI_INFO_INTERNAL"
        ])
          .then((data) => {
            console.info(`[EntryAbility], data: ${ JSON.stringify(data) }`);
          })
          .catch((err: BusinessError) => {
            console.info(`[EntryAbility], err: ${ JSON.stringify(err) }`);
          })
      } catch( err ) {
        console.info(`[EntryAbility], catch err->${ JSON.stringify(err) }`);
      }
    }).catch((err: BusinessError) => {
      console.error(`[EntryAbility] requestEnableNotification failed, code is ${ err.code }, message is ${ err.message }`);
    });
  }

  // 开始长时任务
  startContinuousTask() {
    let context: Context = getContext(this);
    // 通知参数，指定点击长时任务通知后跳转的应用
    let wantAgentInfo: wantAgent.WantAgentInfo = {
      wants : [
        {
          bundleName : (context as common.UIAbilityContext).abilityInfo.bundleName ,
          abilityName : (context as common.UIAbilityContext).abilityInfo.name
        }
      ] ,
      operationType : wantAgent.OperationType.START_ABILITY ,
      requestCode : 0 ,
      wantAgentFlags : [ wantAgent.WantAgentFlags.UPDATE_PRESENT_FLAG ]
    };
    wantAgent.getWantAgent(wantAgentInfo).then((wantAgentObj: WantAgent) => {
      backgroundTaskManager.startBackgroundRunning(context ,
        backgroundTaskManager.BackgroundMode.WIFI_INTERACTION , wantAgentObj).then(() => {
        console.info(`Succeeded in operationing startBackgroundRunning.`);
      }).catch((err: BusinessError) => {
        console.error(`Failed to operation startBackgroundRunning. Code is ${ err.code }, message is ${ err.message }`);
      });
    });
  }

  // 停止长时任务
  stopContinuousTask() {
    backgroundTaskManager.stopBackgroundRunning(getContext()).then(() => {
      console.info(`Succeeded in operationing stopBackgroundRunning.`);
    }).catch((err: BusinessError) => {
      console.error(`Failed to operation stopBackgroundRunning. Code is ${ err.code }, message is ${ err.message }`);
    });
  }

  build() {
    Column() {
      Stack({ alignContent : Alignment.TopStart }) {
        TestImageDisplay({ testItem : this.testItem })
        PageTitle({ testItem : this.testItem })
      }

      Stack().height("1vp").backgroundColor($r("app.color.black"));

      Column() {
        Row() {
          Text("测试次数：").fontSize($r('app.float.font_18')).width(ConfigData.WH_25_100).padding({ left : "10vp" })
          TextInput({ placeholder : "请输入循环次数，默认为30" })
            .placeholderColor(Color.Grey)
            .placeholderFont({ size : $r('app.float.font_16') })
            .caretColor(Color.Blue)
            .width(ConfigData.WH_50_100)
            .fontSize($r('app.float.font_16'))
            .fontColor($r('app.color.title_black_color'))
            .onChange((value: string) => {
              this.testNumbers = parseInt(value);
            })
          Image($r('app.media.Switch'))
            .height(50)
            .padding({ top : 5 , bottom : 5 , left : 5 })
            .width(ConfigData.WH_15_100)
            .onClick(() => {
              this.showList = !this.showList;
            })
        }
      }.width(ConfigData.WH_100_100).height(ConfigData.WH_6_100).backgroundColor($r("app.color.lead"))

      if ( this.showList ) {
        setWifiDevice()
      } else {
        Column() {

          Column() {
            Text("测试结果：")
              .fontSize("22vp")
              .width(ConfigData.WH_95_100)
              .align(Alignment.TopStart)
              .padding({ top : "10vp" })
            Scroll() {
              Column() {
                Text("测试次数：" + this.testNumbers)
                  .fontSize("18vp")
                  .margin({ top : "5vp" , left : "10vp" , right : "5vp" })
                  .textAlign(TextAlign.Start)
                  .width(ConfigData.WH_90_100)
                Text("连接的成功次数：" + this.connectSuccessNumber)
                  .fontSize("18vp")
                  .margin({ top : "5vp" , left : "10vp" , right : "5vp" })
                  .textAlign(TextAlign.Start)
                  .width(ConfigData.WH_90_100)
                Text("连接的失败次数：" + this.connectFailNumber)
                  .fontSize("18vp")
                  .margin({ top : "5vp" , left : "10vp" , right : "5vp" })
                  .textAlign(TextAlign.Start)
                  .width(ConfigData.WH_90_100)
                Text("断连的成功次数：" + this.disconnectSuccessNumber)
                  .fontSize("18vp")
                  .margin({ top : "5vp" , left : "10vp" , right : "5vp" })
                  .textAlign(TextAlign.Start)
                  .width(ConfigData.WH_90_100)
                Text("断连的失败次数：" + this.disconnectFailNumber)
                  .fontSize("18vp")
                  .margin({ top : "5vp" , left : "10vp" , right : "5vp" })
                  .textAlign(TextAlign.Start)
                  .width(ConfigData.WH_90_100)
              }
              .width(ConfigData.WH_95_100)
              .backgroundColor($r("sys.color.ohos_id_color_foreground_contrary"))
            }
            .scrollBarWidth(10)
            .scrollBar(BarState.Auto)

          }.width(ConfigData.WH_100_100).height(ConfigData.WH_35_100)

          Column() {
            Text("log显示：")
              .fontSize("22vp")
              .width(ConfigData.WH_95_100)
              .align(Alignment.TopStart)
              .padding({ top : "10vp" })
            List() {
              ListItem() {
                Text(this.connectionMessageLog)
                  .fontSize("18vp")
                  .margin({ top : "5vp" , left : "10vp" , right : "5vp" })
                  .textAlign(TextAlign.Start)
                  .width(ConfigData.WH_90_100)
              }
            }
            .height(ConfigData.WH_83_100)
            .width(ConfigData.WH_95_100)
            .backgroundColor($r("sys.color.ohos_id_color_foreground_contrary"))
          }.width(ConfigData.WH_100_100).height(ConfigData.WH_60_100)
        }.width(ConfigData.WH_100_100).height(ConfigData.WH_60_100).backgroundColor($r("app.color.lead"))

        Stack().height("1vp").backgroundColor($r("app.color.black"));

        Column() {
          Row() {
            Button("开始连接测试")
              .operateButtonStyle($r('app.color.main_blue'))
              .padding({ right : "10vp" })
              .onClick(async(event: ClickEvent) => {
                console.log(TAG , "start WiFiConnect test-------------")
                this.startContinuousTask();
                await this.forLoop()
              })

            Blank().width(ConfigData.WH_10_100)

            Button("关闭连接测试")
              .operateButtonStyle($r('app.color.main_blue'))
              .padding({ right : "10vp" })
              .onClick(async(event: ClickEvent) => {
                this.connectionLoopState = false
                console.log(TAG , "准备测试停止 -------------")
                this.connectionMessageLog = ""
                this.connectNumber = 0;
                this.stopContinuousTask();
              })
          }
        }.width(ConfigData.WH_100_100).height(ConfigData.WH_25_100).backgroundColor($r("app.color.lead"))
      }
    }.height(ConfigData.WH_100_100).width(ConfigData.WH_100_100).backgroundColor($r("app.color.lead"))
  }

  onConnectionState() {
    console.log(TAG , "on.wifiConnectionChange start--------")
    wifi.on("wifiConnectionChange" , (number) => {
      console.log(TAG , " on.wifiConnectionChange --------")
      let wifiConnectionMessage = ""
      switch ( number ) {
        case 0:
          wifiConnectionMessage = "disconnected";
          break;
        case 1:
          wifiConnectionMessage = "connected";
          break;
        default:
          wifiConnectionMessage = '未知状态';
          break;
      }
      console.log(TAG , "wifi连接状态为:" + wifiConnectionMessage)
      if ( number == 1 ) {
        // 打时间戳
        this.connect_EndTime = new Date().getTime()
        console.log(TAG , "WIFI连接结束时间: " + this.connect_EndTime + "ms")
        this.ipInfo = wifiManager.getIpInfo().ipAddress
      } else if ( number == 0 ) {
        this.disconnect_EndTime = new Date().getTime()
        console.log(TAG , "WIFI断连结束时间: " + this.disconnect_EndTime + "ms")
        let removeMessage = wifi.removeAllNetwork()
        console.log(TAG , "移除网络配置接口的调用结果: " + removeMessage)
      }
      console.log(TAG , " on.wifiConnectionChange callback success --------")
      prompt.showToast({ message : "wifi连接状态: " + wifiConnectionMessage })
      this.connectionMessage = wifiConnectionMessage
      AppStorage.setOrCreate("connectionMessage" , this.connectionMessage)
      console.log(TAG , " on.wifiConnectionChange callback end--------")
      return this.connectionMessage
    })
  }

  aboutToDisappear() {
    wifi.off("wifiConnectionChange")
  }

  async forLoop() {
    this.connectionLoopState = true
    this.connectNumber = 0
    this.connectSuccessNumber = 0
    this.connectFailNumber = 0
    this.disconnectNumber = 0
    this.disconnectSuccessNumber = 0
    this.disconnectFailNumber = 0

    this.connectionMessageLog = ""
    this.onConnectionState() // 监听函数
    let funcMessage = false // 函数信息
    let connectionLoopState = this.connectionLoopState
    let connectNumber = this.connectNumber
    let connect_StartTime = this.connect_StartTime
    let connect_EndTime = this.connect_EndTime
    let connect_SpendTime = this.connect_SpendTime
    let connectSuccessNumber = this.connectSuccessNumber
    let disconnectNumber = this.disconnectNumber
    let disconnect_StartTime = this.disconnect_StartTime
    let disconnect_EndTime = this.disconnect_EndTime
    let disconnect_SpendTime = this.disconnect_SpendTime
    let connectionMessage = this.connectionMessage
    let disconnectSuccessNumber = this.disconnectSuccessNumber
    let disconnectFailNumber = this.disconnectFailNumber
    let connectFailNumber = this.connectFailNumber
    let ipInfo = this.ipInfo
    for ( this.connectNumber ; this.connectNumber < this.testNumbers ; this.connectNumber ++ ) {
      if ( !connectionLoopState ) {
        console.log(TAG , "测试结束------------")
        break;
      } else {
        let wifiState = wifiManager.isWifiActive()
        if ( !wifiState ) {
          wifi.enableWifi()
          console.log(TAG , "wifi当前未使能，已经使能，正常开始测试------")
        } else {
          console.log(TAG , "wifi当前使能，正常开始测试------")
        }
        await sleep(3)

        wifiManager.connectToDevice(this.wifiConfig)
        // 打时间戳
        connect_StartTime = new Date().getTime()
        console.log(TAG , "第" + (connectNumber + 1) + "次WIFI连接-----")
