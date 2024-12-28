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
import { StressContentTable } from '../../Component/stressContentTable';
import router from '@ohos.router';
import ConfigData from '../../utils/ConfigData';
import { initHotspotStressData } from '../../entryability/model/stressTestDataModels'
import { TestData } from '../../entryability/model/testData'
import { FilterTable } from '../../Component/filterTable';
import wifi from '@ohos.wifi';
import { resolveIP } from '../../utils/Util'
import socket from '@ohos.net.socket';

import http from '@ohos.net.http'
import request from '@ohos.request'
import common from '@ohos.app.ability.common';
import abilityAccessCtrl , { Context , PermissionRequestResult } from '@ohos.abilityAccessCtrl';
import { BusinessError } from '@ohos.base';

const TAG = '[Stress]'

const G_NAME = 'DaYu200'

interface DownloadConfig {
  url: string,
  filePath: string
}

interface LocalAddr {
  address: string,
  family: number,
  port: number
}

interface OppositeAddr {
  address: string,
  family: number,
  port: number
}

interface myParams extends Object {
  testItem: TestData
}

let localAddr: LocalAddr = {
  address : resolveIP(wifi.getIpInfo().ipAddress) ,
  family : 1 ,
  port : 9090
}
let oppositeAddr: OppositeAddr = {
  address : '' ,
  family : 1 ,
  port : 0
}
let loginCount = 0

let udp = socket.constructUDPSocketInstance()

/**
 * hotspot StressTest Page Of Wifi test
 */

@Entry
@Component
struct hotspotStressTest {
  @State showList: boolean = true;
  @State showListItems: boolean = false;
  @State showListItem: boolean = false;
  private testItem: TestData = (router.getParams() as myParams).testItem
  @State message: string = "";
  @StorageLink('stressNumber') stressNumber: number = 0;
  @StorageLink('stressTime') stressTime: number = 0;
  @State h_ssid: string = "testApp1";
  @State h_securityType: number = 3;
  @State h_band: number = 2;
  @State h_preSharedKey: string = "12345678";
  @State h_maxConn: number = 3;
  @State h_channel: number = 36;
  @State peripheralDeviceId: string = ''

  aboutToAppear() {
    AppStorage.setOrCreate("h_ssid" , this.h_ssid)
    AppStorage.setOrCreate("h_securityType" , this.h_securityType)
    AppStorage.setOrCreate("h_band" , this.h_band)
    AppStorage.setOrCreate("h_preSharedKey" , this.h_preSharedKey)
    AppStorage.setOrCreate("h_maxConn" , this.h_maxConn)
    AppStorage.setOrCreate("h_channel" , this.h_channel)
  }

  init() {
    console.log("init-----------")
  }

  build() {
    Column() {
      Stack({ alignContent : Alignment.TopStart }) {
        TestImageDisplay({ testItem : this.testItem })
        PageTitle({ testItem : this.testItem })
      }

      Stack().height("0.5vp").backgroundColor("#000000");
      Column() {
        Row() {
          Text("热点IP")
            .fontSize("18vp")
            .height(60)
            .width(ConfigData.WH_20_100)
            .padding({ left : 10 })
          TextInput({ text : this.peripheralDeviceId , placeholder : "input peripheral deviceId." })
            .fontSize("18vp")
            .onChange((strInput: string) => {
              this.peripheralDeviceId = strInput;
              AppStorage.setOrCreate("peripheralDeviceId" , strInput)
            })
            .width(ConfigData.WH_65_100)
          Image($r('app.media.Switch'))
            .height(40)
            .width(40)
            .padding({ top : 10 , bottom : 10 , left : 2 })
            .onClick(() => {
              this.showList = !this.showList;
            })
        }
        .margin({ top : "10vp" , left : "20vp" , right : "20vp" })
        .alignItems(VerticalAlign.Center)
        .backgroundColor($r("app.color.white"))
        .width(ConfigData.WH_100_100)
        .height(60)
      }
      .align(Alignment.Start)

      if ( this.showList ) {
        Column() {
          Scroll() {
            Column() {
              Text("stress测试：")
                .fontSize("17vp")
                .margin({ top : "5vp" , left : "15vp" })
                .textAlign(TextAlign.Start)
                .width(ConfigData.WH_100_100)
              Text("stress测试次数:" + this.stressNumber)
                .fontSize("17vp")
                .margin({ top : "5vp" , left : "15vp" })
                .textAlign(TextAlign.Start)
                .width(ConfigData.WH_90_100)
              Text("stress测试时间：" + this.stressTime + "ms")
                .fontSize("17vp")
                .margin({ top : "5vp" , left : "15vp" })
                .textAlign(TextAlign.Start)
                .width(ConfigData.WH_90_100)
              Flex({ alignItems : ItemAlign.Center , justifyContent : FlexAlign.SpaceBetween }) {
                Button({ type : ButtonType.Normal , stateEffect : true }) {
                  Text('传输数据').fontSize("20vp").fontColor(0xffffff).margin({ left : 2 , right : 2 })
                }
                .borderRadius(8)
                .backgroundColor($r("app.color.blue"))
                .width(100)
                .height(60)
                .margin({ top : "5vp" , left : "10vp" , bottom : "5vp" })
                .align(Alignment.Start)
                .onClick(async(event: ClickEvent) => {
                  this.showListItems = true
                  this.showListItem = false
                })

                Button({ type : ButtonType.Normal , stateEffect : true }) {
                  Row() {
                    Text('上传下载').fontSize("20vp").fontColor(0xffffff).margin({ left : 2 , right : 2 })
                  }.alignItems(VerticalAlign.Center)
                }
                .borderRadius(8)
                .backgroundColor($r("app.color.blue"))
                .width(100)
                .height(60)
                .margin({ top : "5vp" , left : "10vp" , right : "10vp" , bottom : "5vp" })
                .align(Alignment.Center)
                .onClick(async(event: ClickEvent) => {
                  this.showListItem = true
                  this.showListItems = false
                })

                Button({ type : ButtonType.Normal , stateEffect : true }) {
                  Row() {
                    Text('返回接口').fontSize("20vp").fontColor(0xffffff).margin({ left : 2 , right : 2 })
                  }.alignItems(VerticalAlign.Bottom)
                }
                .borderRadius(8)
                .backgroundColor($r("app.color.blue"))
                .width(100)
                .height(60)
                .margin({ top : "5vp" , right : "10vp" , bottom : "5vp" })
                .align(Alignment.End)
                .onClick(async(event: ClickEvent) => {
                  this.showListItem = false
                  this.showListItems = false
                })
              }
              .backgroundColor($r("sys.color.ohos_id_color_sub_background"))
              .width(ConfigData.WH_100_100)
            }
            .width(ConfigData.WH_100_100)
          }
          .scrollBarWidth(10)
          .scrollBar(BarState.Auto)
        }
        .height(130)
        .width(ConfigData.WH_95_100)
        .backgroundColor($r("sys.color.ohos_id_color_foreground_contrary"))

        if ( this.showListItems == true && this.showListItem == false ) {
          socketTest()
        } else if ( this.showListItems == false && this.showListItem == true ) {
          upDownLoad()
        } else {
          StressContentTable({
            testItem : this.testItem , apiItems : initHotspotStressData()
          })
        }
      } else {
        FilterTable({ testItem : this.testItem })
      }
    }
    .alignItems(HorizontalAlign.Center)
    .backgroundColor($r("app.color.lead"))
    .height(ConfigData.WH_100_100)
  }
}

@Component
struct socketTest {
  login_feng: boolean = false
  login_wen: boolean = false
  user: string = ''
  roomDialog: boolean = false
  @State confirmDialog: boolean = false
  @State ipDialog: boolean = true
  @State txtDialog: boolean = true
  @State warnDialog: boolean = false
  @State warnText: string = ''
  roomNumber: string = ''
  @State bindMsg: string = "未绑定"
  @State receiveMsg: string = '待接收数据'

  bindOption() {
    let bindOption = udp.bind(localAddr);
    bindOption.then(() => {
      console.log(TAG , 'bind success')
      this.bindMsg = "绑定成功"
    }).catch((err: BusinessError) => {
      console.log(TAG , 'bind fail' + err)
      this.bindMsg = "绑定失败"
    })
    udp.on('message' , data => {
      console.log(TAG , `data:${ JSON.stringify(data) }`)
      let buffer = data.message
      let dataView = new DataView(buffer)
      console.log(TAG , `length = ${ dataView.byteLength }`)
      let str = ""
      for ( let i = 0 ;i < dataView.byteLength ; ++i ) {
        let c = String.fromCharCode(dataView.getUint8(i))
        if ( c != "\n" ) {
          str += c
        }
      }
      this.receiveMsg = str
      this.confirmDialog = true
    })
  }

  build() {
    Stack({ alignContent : Alignment.Center }) {
      Column() {
        if ( this.ipDialog ) {
          Column() {
            Text('本地IP：' + localAddr.address)
              .fontSize(25)
              .margin({ top : 10 })
            TextInput({ placeholder : '请输入对端ip' })
              .width(200)
              .fontSize(25)
              .margin({ top : 10 })
              .onChange((value: string) => {
                oppositeAddr.address = value
                oppositeAddr.port = 9090
              })

            if ( this.warnDialog ) {
              Text(this.warnText)
                .width(200)
                .fontSize(10)
                .fontColor(Color.Red).margin({ top : 5 })
            }
            Row() {
              Button($r('app.string.confirm'))
                .fontColor(Color.Black)
                .height(40)
                .width(130)
                .fontSize(25)
                .margin({ bottom : 10 })
                .onClick(() => {
                  if ( oppositeAddr.address === '' ) {
                    this.warnDialog = true
                    this.warnText = '请先输入对端IP'
                  } else {
                    this.bindOption()
                    this.txtDialog = true
                    console.log(TAG , `peer ip=${ oppositeAddr.address }`)
                    console.log(TAG , `peer port=${ oppositeAddr.port }`)
                    console.log(TAG , `peer port=${ localAddr.port }`)
                  }
                })
                .backgroundColor(0xffffff)

              Button($r('app.string.send_message'))
                .fontColor(Color.Black)
                .height(40)
                .width(130)
                .fontSize(25)
                .margin({ bottom : 10 })
                .onClick(() => {
                  {
                    //发送数据
                    udp.send({
                      data : 'online' ,
                      address : oppositeAddr
                    }).then(data => {
                      console.log(TAG , `send sucess : ${ data }`)
                    }).catch((error: BusinessError) => {
                      console.log(TAG , `TAG,send : ${ JSON.stringify(error) }`)
                    })

                    console.log(TAG , `send peer ip=${ oppositeAddr.address }`)
                    console.log(TAG , `send peer port=${ oppositeAddr.port }`)
                    console.log(TAG , `send peer port=${ localAddr.port }`)
                  }
                })
                .backgroundColor(0xffffff)
            }

            Text(this.bindMsg)
              .width(200)
              .fontSize(25)
              .fontColor(Color.Red).margin({ top : 3 })
          }
          .width(ConfigData.WH_80_100)
          .height(200)
          .margin({ top : ConfigData.WH_3_100 })
          .backgroundColor(Color.White)
          .border({ radius : 10 , width : 3 })
        }

        if ( this.txtDialog ) {
          Column() {
            Text('接收本地IP：' + localAddr.address)
              .fontSize(25)
              .margin({ top : 10 })

            Text(this.receiveMsg)
              .width(200)
              .fontSize(25)
              .fontColor(Color.Red).margin({ top : 5 })

          }
          .width(ConfigData.WH_80_100)
          .height(180)
          .margin({ top : ConfigData.WH_3_100 })
          .backgroundColor(Color.White)
          .border({ radius : 10 , width : 3 })
        }

      }.width(ConfigData.WH_100_100).height(ConfigData.WH_100_100)
    }
  }
}

@Component
struct upDownLoad {
  @State receivedSize: number = 0
  @State totalSize: number = 0
  @State message: string = ''
  @State files: Array<string> = []
  @State uploads: Array<string> = []
  private downloadConfig: DownloadConfig = {
    url : 'http://192.168.62.127:8000' ,
    filePath : ""
  }

  aboutToAppear() {
    console.log(TAG , `enter Index aboutToAppear`)
    let context: Context = getContext(this) as common.UIAbilityContext;
    let atManager: abilityAccessCtrl.AtManager = abilityAccessCtrl.createAtManager();
    // requestPermissionsFromUser会判断权限的授权状态来决定是否唤起弹窗
    atManager.requestPermissionsFromUser(context , [ 'ohos.permission.INTERNET' ])
      .then((data: PermissionRequestResult) => {
        console.log(TAG , `grantPermission,requestPermissionsFromUser,result.requestCode=${ data }`)
      })
    this.files.push("http://192.168.62.