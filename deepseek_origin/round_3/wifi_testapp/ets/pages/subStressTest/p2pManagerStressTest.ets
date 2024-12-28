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
import { initP2pManagerStressData } from '../../entryability/model/stressTestDataModels'
import { TestData } from '../../entryability/model/testData'
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
 * P2pManager StressTest Page of Wifi Test
 */

@Entry
@Component
struct P2pManagerStressTest {
  @State showList: boolean = false;
  @State showListItems: boolean = false;
  @State showListItem: boolean = false;
  private testItem: TestData = (router.getParams() as myParams).testItem
  @State message: string = "";
  @State deviceId: string = '';
  @State currentClick: number = - 1;
  @State deviceAddressManager: string = '6c:96:d7:3d:87:6f';
  @State netIdManager: number = - 2;
  @State passphraseManager: string = "12345678";
  @State groupNameManager: string = "testGroup"
  @State goBandManager: number = 0;
  @State devNameManager: string = "MyTestDevice"
  @StorageLink('stressNumber') stressNumber: number = 0;
  @StorageLink('stressTime') stressTime: number = 0;
  @StorageLink('pairedMac') pairedMac: string = "";

  aboutToAppear() {
    AppStorage.setOrCreate("deviceAddressManager" , this.deviceAddressManager)
    AppStorage.setOrCreate("netIdManager" , this.netIdManager)
    AppStorage.setOrCreate("passphraseManager" , this.passphraseManager)
    AppStorage.setOrCreate("groupNameManager" , this.groupNameManager)
    AppStorage.setOrCreate("goBandManager" , this.goBandManager)
    AppStorage.setOrCreate("devNameManager" , this.devNameManager)
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
          Text("目标设备IP：")
            .fontSize("20vp")
            .height(40)
            .padding({ top : 5 })
            .width(ConfigData.WH_28_100)
            .align(Alignment.Start)

          TextInput({ text : this.pairedMac , placeholder : "请输入即将连接的设备MAC" })
            .height(40)
            .borderStyle(BorderStyle.Dashed)
            .backgroundColor($r("app.color.white"))
            .onChange((str) => {
              this.deviceId = str;
            })
            .width(ConfigData.WH_45_100)
            .padding({ top : 5 , left : 5 })
          Image($r('app.media.Switch'))
            .height(50)
            .padding({ top : 5 , bottom : 5 , left : 10 })
            .width(ConfigData.WH_15_100)
            .onClick(() => {
              this.showList = !this.showList;
            })
        }
      }
      .align(Alignment.Start)

      if ( this.showList ) {
        p2pManagerSetting()
      } else {
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
          StressContentTable({ testItem : this.testItem , apiItems : initP2pManagerStressData() })
        }
      }
    }
    .alignItems(HorizontalAlign.Center)
    .backgroundColor($r("app.color.lead"))
    .height(ConfigData.WH_100_100)
  }
}

@Component
struct p2pManagerSetting {
  @State deviceAddressManager: string = '6c:96:d7:3d:87:6f';
  @State netIdManager: number = - 2;
  @State passphraseManager: string = "12345678";
  @State groupNameManager: string = "testGroup"
  @State goBandManager: number = 0;
  @State devNameManager: string = "MyTestDevice"

  aboutToAppear() {
    AppStorage.setOrCreate("deviceAddressManager" , this.deviceAddressManager)
    AppStorage.setOrCreate("netIdManager" , this.netIdManager)
    AppStorage.setOrCreate("passphraseManager" , this.passphraseManager)
    AppStorage.setOrCreate("groupNameManager" , this.groupNameManager)
    AppStorage.setOrCreate("goBandManager" , this.goBandManager)
    AppStorage.setOrCreate("devNameManager" , this.devNameManager)
  }

  build() {
    Column() {
      Row() {
        Text("deviceAddressManager:").fontSize(16).width(120)
        TextInput({ text : this.deviceAddressManager , placeholder : '6c:96:d7:3d:87:6f' })
          .fontSize("15vp")
          .onChange((strInput: string) => {
            this.deviceAddressManager = strInput;
            //判断合法性
            if ( strInput.length >= 1 ) {
              AppStorage.setOrCreate('deviceAddressManager' , strInput);
            }
          })
          .width(ConfigData.WH_80_100)
          .borderRadius(1)
      }
      .backgroundColor($r("app.color.moon"))
      .padding(5)
      .justifyContent(FlexAlign.Start)
      .alignItems(VerticalAlign.Center)

      Column() {
        Stack().height("0.25vp").backgroundColor("#000000");
        Column() {
          Row() {
            Text("netIdManager:").fontSize(15).width(60);
            TextInput({ text : this.netIdManager.toString() , placeholder : "-2" })
              .fontSize("15vp")
              .onChange((strInput: string) => {
                this.netIdManager = Number(strInput);
                //判断合法性
                if ( strInput.length >= 1 ) {
                  AppStorage.setOrCreate('netIdManager' , strInput);
                }
              })
              .width(ConfigData.WH_80_100)
              .borderRadius(1)
          }
          .padding(5)
          .justifyContent(FlexAlign.Start)
          .alignItems(VerticalAlign.Center)
          .backgroundColor($r("app.color.spring"))

          Row() {
            Text("passphraseManager:").fontSize(15).width(100);
            TextInput({ text : this.passphraseManager , placeholder : "input passphrase" })
              .fontSize("15vp")
              .onChange((strInput: string) => {
                this.passphraseManager = strInput;
                if ( strInput.length >= 1 ) {
                  AppStorage.setOrCreate('passphraseManager' , strInput);
                }
              })
              .width(ConfigData.WH_80_100)
              .borderRadius(1)
          }
          .backgroundColor($r("app.color.spring"))
          .padding(5)
          .justifyContent(FlexAlign.Start)
          .alignItems(VerticalAlign.Center)

          Row() {
            Text("groupNameManager:").fontSize(15).width(100);
            TextInput({ text : this.groupNameManager , placeholder : "testGroup" })
              .fontSize("15vp")
              .onChange((strInput: string) => {
                this.groupNameManager = strInput;
                if ( strInput.length >= 1 ) {
                  AppStorage.setOrCreate('groupNameManager' , strInput);
                }
              })
              .width(ConfigData.WH_80_100)
              .borderRadius(1)
          }
          .backgroundColor($r("app.color.spring"))
          .padding(5)
          .justifyContent(FlexAlign.Start)
          .alignItems(VerticalAlign.Center)

          Row() {
            Text("goBandManager:").fontSize(15).width(80);
            TextInput({ text : this.goBandManager.toString() , placeholder : "0" })
              .fontSize("15vp")
              .onChange((strInput: string) => {
                this.goBandManager = Number(strInput);
                if ( strInput.length >= 1 ) {
                  AppStorage.setOrCreate('goBandManager' , strInput);
                }
              })
              .width(ConfigData.WH_80_100)
              .borderRadius(1)
          }
          .backgroundColor($r("app.color.spring"))
          .padding(5)
          .justifyContent(FlexAlign.Start)
          .alignItems(VerticalAlign.Center)

          Row() {
            Text("devNameManager:").fontSize(15).width(100);
            TextInput({ text : this.devNameManager , placeholder : "MyTestDevice" })
              .fontSize("15vp")
              .onChange((strInput: string) => {
                this.devNameManager = strInput;
                if ( strInput.length >= 1 ) {
                  AppStorage.setOrCreate('devNameManager' , strInput);
                }
              })
              .width(ConfigData.WH_80_100)
              .borderRadius(1)
          }
          .backgroundColor($r("app.color.spring"))
          .padding(5)
          .justifyContent(FlexAlign.Start)
          .alignItems(VerticalAlign.Center)
        }

        Stack().height("0.25vp").backgroundColor("#000000");
      }
    }
  }
}

@Component
struct socketTest {
  @State login_feng: boolean = false
  @State login_wen: boolean = false
  @State user: string = ''
  @State roomDialog: boolean = false
  @State confirmDialog: boolean = false
  ipDialog: boolean = true
  @State txtDialog: boolean = true
  @State warnDialog: boolean = false
  @State warnText: string = ''
  roomNumber: string = ''
  @State bindMsg: string = "未绑定"
  @State receiveMsg: string = '待接收数据'

  bindOption() {
    let bindOption = udp.bind(localAddr)
    bindOption.then(() => {
      console.log(TAG , 'bind success')
      this.bindMsg = "绑定成功"
    }).catch((err: BusinessError