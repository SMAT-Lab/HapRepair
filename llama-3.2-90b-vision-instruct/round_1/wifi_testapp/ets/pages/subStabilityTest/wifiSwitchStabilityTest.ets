
import { PageTitle } from '../../Component/pageTitle';
import { TestImageDisplay } from '../../Component/testImageDisplay';
import router from '@ohos.router';
import ConfigData from '../../Utils/ConfigData';
import { TestData } from '../../entryability/model/testData';
import wifi from '@ohos.wifi';
import wifiManager from '@ohos.wifiManager';
import prompt from '@system.prompt';
import fs from '@ohos.file.fs';
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
 * Custom button style.
 */
@Extend(Button)
function operateButtonStyle(color: Resource) {
  .width($r('app.float.button_width'))
  .height($r('app.float.button_height'))
  .fontSize($r('app.float.font_16'))
  .fontWeight(500)
  .fontColor(color)
  .backgroundColor($r('app.color.button_background'))
}

/**
 * WiFiSwitchStabilityTest Page Of wifi test
 */
const TAG = '[WiFiSwitchStabilityTest]'

@Entry
@Component
struct WiFiSwitchStabilityTest {
  private testItem: TestData = (router.getParams() as myParams).testItem
  @State open_StartTime: number = 0
  @State open_EndTime: number = 0
  @State open_SpendTime: number = 0
  @State close_StartTime: number = 0
  @State close_EndTime: number = 0
  @State close_SpendTime: number = 0
  @State openWifiNumber: number = 0 // 打开WiFi的次数
  @State closeWifiNumber: number = 0 // 关闭WiFi的次数
  @State openSuccessNumber: number = 0 // 打开WiFi的成功次数
  @State openFailNumber: number = 0 // 打开WiFi的失败次数
  @State closeSuccessNumber: number = 0 // 关闭WiFi的成功次数
  @State closeFailNumber: number = 0 // 关闭WiFi的失败次数
  @State message: string = "测试结果:"
  @State testNumbers: number = 30 //测试次数
  @State wifiState: boolean = wifiManager.isWifiActive()
  @State testNumbersValue: number = 30
  @State file!: number
  @State filePathDir!: string
  private file!: fs.File
  @StorageLink("pathDir") pathDir: string = ""
  @State stateMessage: string = ""
  @State stateMessageLog: string = ""
  @State switchLoopState: Boolean = true

  async openLogFile() {
    let time: number = new Date().getTime()
    let currentTime: string = timestampToDate(time)
    console.log(TAG , "当前时间: " + currentTime)
    this.filePath = this.pathDir + ("/" + currentTime + ".txt");
    console.log(TAG , "沙箱路径为: " + this.filePath)
    this.file = fs.openSync(this.filePath , fs.OpenMode.READ_WRITE | fs.OpenMode.CREATE);
    AppStorage.setOrCreate("fsFile" , this.file)
    AppStorage.setOrCreate("openWifiNumber" , this.openWifiNumber)
    console.log(TAG , "数据文件的沙箱路径为: " + this.filePath)
    console.log(TAG , "file fd: " + this.file.fd);
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
          TextInput({ placeholder : "请输入循环次数，默认为：30" })
            .placeholderColor(Color.Grey)
            .placeholderFont({ size : $r('app.float.font_16') })
            .caretColor(Color.Blue)
            .width(ConfigData.WH_70_100)
            .fontSize($r('app.float.font_16'))
            .fontColor($r('app.color.title_black_color'))
            .onChange((value: string) => {
              this.testNumbers = parseInt(value);
            })
        }

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
              Text("打开WiFi的成功次数：" + this.openSuccessNumber)
                .fontSize("18vp")
                .margin({ top : "5vp" , left : "10vp" , right : "5vp" })
                .textAlign(TextAlign.Start)
                .width(ConfigData.WH_90_100)
              Text("打开WiFi的失败次数：" + this.openFailNumber)
                .fontSize("18vp")
                .margin({ top : "5vp" , left : "10vp" , right : "5vp" })
                .textAlign(TextAlign.Start)
                .width(ConfigData.WH_90_100)
              Text("关闭WiFi的成功次数：" + this.closeSuccessNumber)
                .fontSize("18vp")
                .margin({ top : "5vp" , left : "10vp" , right : "5vp" })
                .textAlign(TextAlign.Start)
                .width(ConfigData.WH_90_100)
              Text("关闭WiFi的失败次数：" + this.closeFailNumber)
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
        }.width(ConfigData.WH_100_100).height(ConfigData.WH_30_100)

        Column() {
          Text("log显示：")
            .fontSize("22vp")
            .height(ConfigData.WH_15_100)
            .width(ConfigData.WH_95_100)
            .align(Alignment.TopStart)
            .padding({ top : "18vp" })
          List() {
            ListItem() {
              Text(this.stateMessageLog)
                .fontSize("18vp")
                .margin({ top : "5vp" , left : "20vp" , right : "5vp" })
                .textAlign(TextAlign.Start)
                .width(ConfigData.WH_90_100)
            }
          }
          .height(ConfigData.WH_83_100)
          .width(ConfigData.WH_95_100)
          .backgroundColor($r("sys.color.ohos_id_color_foreground_contrary"))
        }.width(ConfigData.WH_100_100).height(ConfigData.WH_60_100)

      }.width(ConfigData.WH_100_100).height(ConfigData.WH_67_100).backgroundColor($r("app.color.lead"))

      Stack().height("1vp").backgroundColor($r("app.color.black"));

      Column() {
        Row() {
          Button("开始WiFi测试")
            .operateButtonStyle($r('app.color.main_blue'))
            .padding({ right : "10vp" })
            .onClick(async(event: ClickEvent) => {
              console.log(TAG , "start WiFiSwitch test-------------")
              this.startContinuousTask();
              await this.openLogFile()
              await this.forLoop()
            })

          Blank().width(ConfigData.WH_10_100)

          Button("关闭WiFi测试")
            .operateButtonStyle($r('app.color.main_blue'))
            .padding({ right : "10vp" })
            .onClick(async(event: ClickEvent) => {
              this.switchLoopState = false
              console.log(TAG , "准备测试停止 -------------")
              this.stateMessageLog = ""
              this.openWifiNumber = 0
              AppStorage.setOrCreate("openWifiNumber" , this.openWifiNumber)
              fs.closeSync(this.file);
              this.stopContinuousTask();
            })
        }
      }.width(ConfigData.WH_100_100).height(ConfigData.WH_25_100).backgroundColor($r("app.color.lead"))
    }.height(ConfigData.WH_100_100).width(ConfigData.WH_100_100).backgroundColor($r("app.color.lead"))
  }

  onWifiState() {
    console.log(TAG , "on.wifiStateChange start--------")
    wifi.on("wifiStateChange" , (number) => {
      console.log(TAG , " on.wifiStateChange --------")
      let wifiStateMessage = ""
      switch ( number ) {
        case 0:
          wifiStateMessage = "inactive";
          break;
        case 1:
          wifiStateMessage = "active";
          break;
        case 2:
          wifiStateMessage = "activating";
          break;
        case 3:
          wifiStateMessage = "de-activating";
          break;
        default:
          wifiStateMessage = '未知状态';
          break;
      }
      console.log(TAG , "WiFi开关状态为:" + wifiStateMessage)
      if ( number == 1 ) {
        // 打时间戳
        this.open_EndTime = new Date().getTime()
        console.log(TAG , "WIFI使能结束时间: " + this.open_EndTime + "ms")
      } else if ( number == 0 ) {
        this.close_EndTime = new Date().getTime()
        console.log(TAG , "WIFI去使能结束时间: " + this.close_EndTime + "ms")
      }
      console.log(TAG , " on.wifiStateChange callback success --------")
      prompt.showToast({ message : "wifi状态: " + wifiStateMessage })
      this.stateMessage = wifiStateMessage
      AppStorage.setOrCreate("stateMessage" , this.stateMessage)
      console.log(TAG , " on.wifiStateChange callback end--------")
      return this.stateMessage
    })
  }

  aboutToDisappear() {
    wifi.off("wifiStateChange")
  }

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
      if ( !this.switchLoopState ) {
        openWifiNumber = 0; // 移除这一行，避免多次的界面刷新
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
        stateMessageLog += "第" + (openWifiNumber + 1) + "次WIFI使能接口的调用结果：" + funcMessage + "\n"
        console.log(TAG , "第" + (openWifiNumber + 1) + "次WIFI使能接口的调用结果：" + funcMessage)
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
}

export function timestampToDate(timestamp: number): string {
  const date = new Date(timestamp);
  let year = date.getFullYear();
  let month: string | number = date.getMonth() + 1;
  month = month >= 10 ? month : ('0' + month);
  let day: string | number = date.getDate();
  day = day >= 10 ? day : ('0' + day);
  let hours: string | number = date.getHours();
  hours = hours >= 10 ? hours : ('0' + hours);
  let minutes: string | number = date.getMinutes();
  minutes = minutes >= 10 ? minutes : ('0' + minutes);
  let seconds: string | number = date.getSeconds();
  seconds = seconds >= 10 ? seconds : ('0' + seconds);
  return `${ year }${ month }${ day }_${ hours }${ minutes }${ seconds }`;
}

async function sleep(time: number): Promise<void> {
  return new Promise<void>((resolve , reject) => {
    setTimeout(() => {
      resolve();
    } , time * 1000);
  });
}