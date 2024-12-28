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

import router from '@ohos.router'
import { PageTitle } from '../../Component/pageTitle'
import { TestImageDisplay } from '../../Component/testImageDisplay';
import { TestData } from '../../entryability/model/testData'
import ConfigData from "../../utils/ConfigData"
import wifi from '@ohos.wifi';
import { resolveIP } from '../../utils/Util'
import socket from '@ohos.net.socket';
import http from '@ohos.net.http'
import request from '@ohos.request'
import common from '@ohos.app.ability.common';
import abilityAccessCtrl , { Context , PermissionRequestResult } from '@ohos.abilityAccessCtrl';
import { BusinessError } from '@ohos.base';

const TAG = '[Benchmark]'

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
 *  classic wifi benchmark Test Page of wifi test
 */

@Entry
@Component
struct WifiBenchmarkTest {
  @State showList: boolean = false;
  message: string = 'WifiBenchmarkTest';
  private testItem: TestData = (router.getParams() as myParams).testItem
  changeIndex: number = - 1
  @StorageLink("wifiBenchmarkTestMessage") wifiBenchmarkTestMessage: string = ""
  w_ssid: string = "TP-LINK_6365";
  w_bssid: string = "6C:B1:58:75:63:65";
  w_preSharedKey: string = "12345678";
  w_isHiddenSsid: boolean = false;
  w_securityType: number = 3
  w_creatorUid: number = 1;
  w_disableReason: number = 0;
  w_netId: number = 100;
  w_randomMacType: number = 0;
  w_randomMacAddr: string = "08:fb:ea:1b:38:aa"
  w_ipType: number = 1;
  w_staticIp_ipAddress: number = 3232235880;
  w_staticIp_gateway: number = 3232235777;
  w_staticIp_dnsServers: number = 3716386629;
  w_staticIp_domains: Array<string> = [ "0", "1", "2" ];
  w_networkId: number = 0; // id,netId,networkId是一个值嘛？
  w_featureId: number = 1;
  receivedSize: number = 0
  totalSize: number = 0
  @State files: Array<string> = []
  @State uploads: Array<string> = []
  @StorageLink('wifiBenchmarkTime') wifiBenchmarkTime: number = 0;

  aboutToAppear() {
    AppStorage.setOrCreate('w_ssid' , this.w_ssid);
    AppStorage.setOrCreate('w_bssid' , this.w_bssid);
    AppStorage.setOrCreate('w_preSharedKey' , this.w_preSharedKey);
    AppStorage.setOrCreate('w_isHiddenSsid' , this.w_isHiddenSsid);
    AppStorage.setOrCreate('w_securityType' , this.w_securityType);
    AppStorage.setOrCreate('w_creatorUid' , this.w_creatorUid);
    AppStorage.setOrCreate('w_disableReason' , this.w_disableReason);
    AppStorage.setOrCreate('w_netId' , this.w_netId);
    AppStorage.setOrCreate('w_randomMacType' , this.w_randomMacType);
    AppStorage.setOrCreate('w_randomMacAddr' , this.w_randomMacAddr);
    AppStorage.setOrCreate('w_ipType' , this.w_ipType);
    AppStorage.setOrCreate('w_staticIp_ipAddress' , this.w_staticIp_ipAddress);
    AppStorage.setOrCreate('w_staticIp_gateway' , this.w_staticIp_gateway);
    AppStorage.setOrCreate('w_staticIp_dnsServers' , this.w_staticIp_dnsServers);
    AppStorage.setOrCreate('w_staticIp_domains' , this.w_staticIp_domains);
    AppStorage.setOrCreate('w_networkId' , this.w_networkId);
    AppStorage.setOrCreate('w_featureId' , this.w_featureId);
  }

  build() {
    Column() {
      Stack({ alignContent : Alignment.TopStart }) {
        TestImageDisplay({ testItem : this.testItem })
        PageTitle({ testItem : this.testItem })
      }

      Stack().height("1vp").backgroundColor("#000000");

      Column() {
        Scroll() {
          Column() {
            Text("性能测试: 测试wifi连接时的数据收发性能")
              .fontSize("18vp")
              .margin({ top : "5vp" , bottom : "5vp" , left : "5vp" })
              .textAlign(TextAlign.Start)
              .width(ConfigData.WH_100_100)
          }
          .width(ConfigData.WH_100_100)
          .height(40)
        }
        .scrollBarWidth(10)
        .scrollBar(BarState.Auto)

        Stack().height("1vp").backgroundColor("#000000")

        Column() {
          Text("性能测试时间：" + this.wifiBenchmarkTime + "ms")
            .fontSize("20vp")
            .margin({ top : "5vp" , left : "10vp" , bottom : "5vp" })
            .textAlign(TextAlign.Start)
            .width(ConfigData.WH_90_100)
            .backgroundColor($r("sys.color.ohos_id_color_sub_background"))

          Flex({ alignItems : ItemAlign.Center , justifyContent : FlexAlign.SpaceBetween }) {
            Button({ type : ButtonType.Normal , stateEffect : true }) {
              Text('传输数据').fontSize("24vp").fontColor(0xffffff).margin({ left : 5 , right : 5 })
            }
            .borderRadius(8)
            .backgroundColor($r("app.color.blue"))
            .width(120)
            .height(60)
            .margin({ top : "5vp" , left : "50vp" , bottom : "5vp" })
            .align(Alignment.Start)
            .onClick(async(event: ClickEvent) => {
              this.showList = false
            })

            Button({ type : ButtonType.Normal , stateEffect : true }) {
              Row() {
                Text('上传下载').fontSize("24vp").fontColor(0xffffff).margin({ left : 5 , right : 5 })
              }.alignItems(VerticalAlign.Center)
            }
            .borderRadius(8)
            .backgroundColor($r("app.color.blue"))
            .width(120)
            .height(60)
            .margin({ top : "5vp" , right : "50vp" , bottom : "5vp" })
            .align(Alignment.Start)
            .onClick(async(event: ClickEvent) => {
              this.showList = true;
            })

          }
          .backgroundColor($r("sys.color.ohos_id_color_sub_background"))
          .width(ConfigData.WH_100_100)
        }
        .backgroundColor($r("sys.color.ohos_id_color_sub_background"))
        .width(ConfigData.WH_100_100)
        .height(100)

        Stack().height("1vp").backgroundColor("#000000")

        Column() {
          if ( !this.showList ) {
            socketTest()
          } else {
            upDownLoad()
          }
        }
        .backgroundColor($r("sys.color.ohos_id_color_sub_background"))
        .width(ConfigData.WH_100_100)
        .height(ConfigData.WH_100_100);
      }
    }
  }
}

@Component
struct socketTest {
  login_feng: boolean = false
  login_wen: boolean = false
  user: string = ''
  roomDialog: boolean = false
  confirmDialog: boolean = false
  ipDialog: boolean = true
  txtDialog: boolean = true
  warnDialog: boolean = false
  warnText: string = ''
  roomNumber: string = ''
  @State bindMsg: string = "未绑定"
  @State receiveMsg: string = '待接收数据'

  bindOption() {
    let bindOption = udp.bind(localAddr)
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
          .height(200)
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
  receivedSize: number = 0
  totalSize: number = 0
  message: string = ''
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
    this.files.push("http://192.168.62.127:8000")
    let httpRequest = http.createHttp()
    httpRequest.request(
      'http://192.168.0.123/?tpl=list&folders-filter=&recursive' ,
      {
        method : http.RequestMethod.GET ,
        header : {
          'Content-Type' : 'text/plain'
        }
      } , (err , data) => {
      console.log(TAG , `error = ${ JSON.stringify(err) }`)
      if ( !err ) {
        console.log(TAG , `data = ${ JSON.stringify(data) }`)
        let result