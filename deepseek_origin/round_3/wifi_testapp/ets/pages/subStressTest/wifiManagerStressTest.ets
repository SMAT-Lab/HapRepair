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
import { initWifiManagerStressData } from '../../entryability/model/stressTestDataModels'
import { TestData , Category } from '../../entryability/model/testData'
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
 * wifiManager StressTest Page Of Wifi test
 */

@Entry
@Component
struct wifiManagerStressTest {
  @State showList: boolean = false;
  @State showListItems: boolean = false;
  @State showListItem: boolean = false;
  private testItem: TestData = (router.getParams() as myParams).testItem
  message: string = "";
  deviceId: string = '';
  currentClick: number = - 1;
  @StorageLink('pairedMac') pairedMac: string = "";
  w_ssid1: string = "TP-LINK_6365";
  w_bssid1: string = "6C:B1:58:75:63:65";
  w_preSharedKey1: string = "12345678";
  w_isHiddenSsid1: boolean = false;
  w_securityType1: number = 3
  w_creatorUid1: number = 1;
  w_disableReason1: number = 0;
  w_netId1: number = 0;
  w_randomMacType1: number = 0;
  w_randomMacAddr1: string = "08:fb:ea:1b:38:aa"
  w_ipType1: number = 1;
  w_staticIp_ipAddress1: number = 3232235880;
  w_staticIp_gateway1: number = 3232235777;
  w_staticIp_dnsServers1: number = 3716386629;
  w_staticIp_domains1: Array<string> = [ "0", "1", "2" ];
  w_networkIdM: number = 0;
  w_featureIdM: number = 1;
  @StorageLink('stressNumber') stressNumber: number = 0;
  @StorageLink('stressTime') stressTime: number = 0;

  aboutToAppear() {
    AppStorage.setOrCreate('w_ssid1' , this.w_ssid1);
    AppStorage.setOrCreate('w_bssid1' , this.w_bssid1);
    AppStorage.setOrCreate('w_preSharedKey1' , this.w_preSharedKey1);
    AppStorage.setOrCreate('w_isHiddenSsid1' , this.w_isHiddenSsid1);
    AppStorage.setOrCreate('w_securityType1' , this.w_securityType1);
    AppStorage.setOrCreate('w_creatorUid1' , this.w_creatorUid1);
    AppStorage.setOrCreate('w_disableReason1' , this.w_disableReason1);
    AppStorage.setOrCreate('w_netId1' , this.w_netId1);
    AppStorage.setOrCreate('w_randomMacType1' , this.w_randomMacType1);
    AppStorage.setOrCreate('w_randomMacAddr1' , this.w_randomMacAddr1);
    AppStorage.setOrCreate('w_ipType1' , this.w_ipType1);
    AppStorage.setOrCreate('w_staticIp_ipAddress1' , this.w_staticIp_ipAddress1);
    AppStorage.setOrCreate('w_staticIp_gateway1' , this.w_staticIp_gateway1);
    AppStorage.setOrCreate('w_staticIp_dnsServers1' , this.w_staticIp_dnsServers1);
    AppStorage.setOrCreate('w_staticIp_domains1' , this.w_staticIp_domains1);
    AppStorage.setOrCreate('w_networkIdM' , this.w_networkIdM);
    AppStorage.setOrCreate('w_featureIdM' , this.w_featureIdM);
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
        WifiManagerDeviceSetting()
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
          StressContentTable({
            testItem : this.testItem , apiItems : initWifiManagerStressData()
          })
        }
      }
    }
    .alignItems(HorizontalAlign.Center)
    .backgroundColor($r("app.color.lead"))
    .height(ConfigData.WH_100_100)
  }
}

@Component
struct WifiManagerDeviceSetting {
  private w_ssid1 = "TP-LINK_6365";
  private w_bssid1 = "6C:B1:58:75:63:65";
  private w_preSharedKey1 = "12345678";
  private w_isHiddenSsid1 = "false";
  w_securityType1: number = 3
  w_creatorUid1: number = 1;
  w_disableReason1: number = 0;
  w_netId1: number = 0;
  w_randomMacType1: number = 0;
  w_randomMacAddr1: string = "08:fb:ea:1b:38:aa"
  w_ipType1: number = 1;
  w_staticIp_ipAddress1: number = 3232235880;
  w_staticIp_gateway1: number = 3232235777;
  w_staticIp_dnsServers1: number = 3716386629;
  w_staticIp_domains1: Array<string> = [ "0", "1", "2" ];
  w_networkIdM: number = 0;
  w_featureIdM: number = 1;

  aboutToAppear() {
    AppStorage.setOrCreate('w_ssid1' , this.w_ssid1);
    AppStorage.setOrCreate('w_bssid1' , this.w_bssid1);
    AppStorage.setOrCreate('w_preSharedKey1' , this.w_preSharedKey1);
    AppStorage.setOrCreate('w_isHiddenSsid1' , this.w_isHiddenSsid1);
    AppStorage.setOrCreate('w_securityType1' , this.w_securityType1);
    AppStorage.setOrCreate('w_creatorUid1' , this.w_creatorUid1);
    AppStorage.setOrCreate('w_disableReason1' , this.w_disableReason1);
    AppStorage.setOrCreate('w_netId1' , this.w_netId1);
    AppStorage.setOrCreate('w_randomMacType1' , this.w_randomMacType1);
    AppStorage.setOrCreate('w_randomMacAddr1' , this.w_randomMacAddr1);
    AppStorage.setOrCreate('w_ipType1' , this.w_ipType1);
    AppStorage.setOrCreate('w_staticIp_ipAddress1' , this.w_staticIp_ipAddress1);
    AppStorage.setOrCreate('w_staticIp_gateway1' , this.w_staticIp_gateway1);
    AppStorage.setOrCreate('w_staticIp_dnsServers1' , this.w_staticIp_dnsServers1);
    AppStorage.setOrCreate('w_staticIp_domains1' , this.w_staticIp_domains1);
    AppStorage.setOrCreate('w_networkIdM' , this.w_networkIdM);
    AppStorage.setOrCreate('w_featureIdM' , this.w_featureIdM);
  }

  build() {
    Column() {
      Row() {
        Text("ssid1:").fontSize("17vp").width(70)
        TextInput({ text : this.w_ssid1 , placeholder : "TP-LINK_6365" })
          .fontSize("15vp")
          .onChange((strInput: string) => {
            this.w_ssid1 = strInput;
            AppStorage.setOrCreate('w_ssid1' , strInput);
          })
          .width(ConfigData.WH_80_100)
          .borderRadius(1)
      }
      .backgroundColor($r("app.color.moon"))
      .padding(5)
      .justifyContent(FlexAlign.Start)
      .alignItems(VerticalAlign.Center)

      Scroll() {
        Column() {
          Stack().height("0.25vp").backgroundColor("#000000");
          Column() {
            Row() {
              Text("bssid1:").fontSize("15vp").width(60);
              TextInput({ text : this.w_bssid1 , placeholder : "6c:b1:58:75:63:67" })
                .fontSize("15vp")
                .onChange((strInput: string) => {
                  this.w_bssid1 = strInput;
                  AppStorage.setOrCreate('w_bssid1' , strInput);
                })
                .width(ConfigData.WH_80_100)
                .borderRadius(1)
            }
            .padding(3)
            .justifyContent(Flex