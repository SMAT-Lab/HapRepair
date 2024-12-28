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
import { ContentTable } from '../../Component/contentTable';
import router from '@ohos.router';
import ConfigData from '../../utils/ConfigData';
import { initWifiManagerApiData } from '../../entryability/model/testDataModels'
import { TestData } from '../../entryability/model/testData'
import wifiManager from '@ohos.wifiManager';

interface myParams extends Object {
  testItem: TestData
}

const TAG = '[WifiManager]'

/**
 * wifiManager Test Page Of Wifi test
 */
interface wifiDeviceConfigManager {
  ssid: string,
  bssid: string,
  preSharedKey: string,
  isHiddenSsid: boolean,
  securityType: wifiManager.WifiSecurityType,
  creatorUid: number,
  disableReason: number,
  netId: number,
  randomMacType: number,
  randomMacAddr: string,
  ipType: wifiManager.IpType,
  staticIp: wifiManager.IpConfig
}

@Component
struct WifiManagerDeviceSetting {
  @StorageLink('isAllowed') isAllowed: number = 1;
  @StorageLink('w_networkIdM') w_networkIdM: number = 0; // id,netId,networkId是一个值嘛？
  @StorageLink('w_featureIdM') w_featureIdM: number = 1;
  @StorageLink('wifiDeviceConfigManager') wifiDeviceConfigManager: wifiManager.WifiDeviceConfig = {
    ssid : "TP-LINK_6365" ,
    bssid : "6C:B1:58:75:63:65" ,
    preSharedKey : "12345678" ,
    isHiddenSsid : false ,
    securityType : 3 ,
    creatorUid : 1 ,
    disableReason : 0 ,
    netId : 1 ,
    randomMacType : 0 ,
    randomMacAddr : "08:fb:ea:1b:38:aa" ,
    ipType : 1 ,
    staticIp : {
      ipAddress : 3232235880 ,
      gateway : 3232235777 ,
      prefixLength : 0 ,
      dnsServers : [ 3716386629 ] ,
      domains : [ "0", "1", "2" ] ,
    }
  }

  aboutToAppear() {
  }

  build() {
    Column() {
      Row() {
        Text("SetScanAlwaysAllowed:").fontSize("17vp").width(200)
        TextInput({ text : this.isAllowed.toString() , placeholder : "1" })
          .fontSize("15vp")
          .onChange((strInput: string) => {
            //判断合法性
            if ( strInput.length >= 1 ) {
              this.isAllowed = Number(strInput);
            }
          })
          .width(ConfigData.WH_50_100)
          .borderRadius(1)
      }
      .backgroundColor($r("app.color.moon"))
      .padding(5)
      .justifyContent(FlexAlign.Start)
      .alignItems(VerticalAlign.Center)

      Row() {
        Text("ssidM:").fontSize("17vp").width(70)
        TextInput({ text : this.wifiDeviceConfigManager.ssid , placeholder : "TP-LINK_6365" })
          .fontSize("15vp")
          .onChange((strInput: string) => {
            //判断合法性
            if ( strInput.length >= 1 ) {
              this.wifiDeviceConfigManager.ssid = strInput;
            }
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
              Text("bssidM:").fontSize("15vp").width(60);
              TextInput({ text : this.wifiDeviceConfigManager.bssid , placeholder : "6c:b1:58:75:63:67" })
                .fontSize("15vp")
                .onChange((strInput: string) => {
                  //判断合法性
                  if ( strInput.length >= 1 ) {
                    this.wifiDeviceConfigManager!.bssid = strInput;
                  }
                })
                .width(ConfigData.WH_80_100)
                .borderRadius(1)
            }
            .padding(3)
            .justifyContent(FlexAlign.Start)
            .alignItems(VerticalAlign.Center)
            .backgroundColor($r("app.color.spring"))

            Row() {
              Text("preSharedKeyM:").fontSize("15vp").width(110);
              TextInput({ text : this.wifiDeviceConfigManager.preSharedKey , placeholder : "12345678" })
                .fontSize("15vp")
                .onChange((strInput: string) => {
                  //判断合法性
                  if ( strInput.length >= 1 ) {
                    this.wifiDeviceConfigManager!.preSharedKey = strInput;
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
              Text("isHiddenSsidM:").fontSize(15).width(110);
              TextInput({ text : String(this.wifiDeviceConfigManager.isHiddenSsid) , placeholder : "false" })
                .fontSize("15vp")
                .onChange((strInput: string) => {
                  //判断合法性
                  if ( strInput.length >= 1 ) {
                    this.wifiDeviceConfigManager!.isHiddenSsid = strInput === 'true';
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
              Text("securityTypeM:").fontSize(15).width(100);
              Column() {
                Select([
                  { value : 'WIFI_SEC_TYPE_INVALID' },
                  { value : 'WIFI_SEC_TYPE_OPEN' },
                  { value : 'WIFI_SEC_TYPE_WEP' },
                  { value : 'WIFI_SEC_TYPE_PSK' },
                  { value : 'WIFI_SEC_TYPE_SAE' }
                ])
                  .selected(3)
                  .value('WIFI_SEC_TYPE_PSK')
                  .font({ size : 17 })
                  .selectedOptionFont({ size : 17 })
                  .optionFont({ size : 15 })
                  .width(ConfigData.WH_80_100)
                  .onSelect((index: number) => {
                    console.log(TAG , "Select:" + index)
                    this.wifiDeviceConfigManager!.securityType = index;
                  })
              }
              .width(ConfigData.WH_80_100)
              .borderRadius(1)
            }
            .backgroundColor($r("app.color.spring"))
            .padding(5)
            .justifyContent(FlexAlign.Start)
            .alignItems(VerticalAlign.Center)

            Row() {
              Text("creatorUidM:").fontSize(15).width(100);
              TextInput({ text : this.wifiDeviceConfigManager?.creatorUid?.toString() ?? "" , placeholder : "1" })
                .fontSize("15vp")
                .onChange((strInput: string) => {
                  //判断合法性
                  if ( strInput.length >= 1 ) {
                    this.wifiDeviceConfigManager!.creatorUid = Number(strInput);
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
              Text("disableReasonM:").fontSize(15).width(120);
              TextInput({ text : this.wifiDeviceConfigManager?.disableReason?.toString() ?? "" , placeholder : "0" })
                .fontSize("15vp")
                .onChange((strInput: string) => {
                  //判断合法性
                  if ( strInput.length >= 1 ) {
                    this.wifiDeviceConfigManager!.disableReason = Number(strInput);
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
              Text("netIdM:").fontSize(15).width(60);
              TextInput({ text : this.wifiDeviceConfigManager?.netId?.toString() ?? "" , placeholder : "1" })
                .fontSize("15vp")
                .onChange((strInput: string) => {
                  //判断合法性
                  if ( strInput.length >= 1 ) {
                    this.wifiDeviceConfigManager!.netId = Number(strInput);
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
              Text("randomMacTypeM:").fontSize(15).width(140);
              TextInput({ text : this.wifiDeviceConfigManager?.randomMacType?.toString() ?? "" , placeholder : "0" })
                .fontSize("15vp")
                .onChange((strInput: string) => {
                  //判断合法性
                  if ( strInput.length >= 1 ) {
                    this.wifiDeviceConfigManager!.randomMacType = Number(strInput);
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
              Text("randomMacAddrM:").fontSize(15).width(140);
              TextInput({ text : this.wifiDeviceConfigManager.randomMacAddr , placeholder : "08:fb:ea:1b:38:aa" })
                .fontSize("15vp")
                .onChange((strInput: string) => {
                  //判断合法性
                  if ( strInput.length >= 1 ) {
                    this.wifiDeviceConfigManager!.randomMacAddr = strInput;
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
              Text("ipTypeM:").fontSize(15).width(60);
              TextInput({ text : this.wifiDeviceConfigManager?.ipType?.toString() ?? '' , placeholder : "1" }) //DHCP
                .fontSize("15vp")
                .onChange((strInput: string) => {
                  //判断合法性
                  if ( strInput.length >= 1 ) {
                    this.wifiDeviceConfigManager!.ipType = Number(strInput);
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
              Text("staticIp_ipAddressM:").fontSize(15).width(135);
              TextInput({
                text : this.wifiDeviceConfigManager?.staticIp?.ipAddress?.toString() ?? '' ,
                placeholder : "3232235880"
              })
                .fontSize("15vp")
                .onChange((strInput: string) => {
                  //判断合法性
                  if ( strInput.length >= 1 ) {
                    this.wifiDeviceConfigManager!.staticIp!.ipAddress = Number(strInput);
                  }
                })
                .width(ConfigData.WH_80_100)
                .borderRadius(1)
            }
            .backgroundColor($r("app.color.moon"))
            .padding(5)
            .justifyContent(FlexAlign.Start)
            .alignItems(VerticalAlign.Center)

            Row() {
              Text("staticIp_gatewayM:").fontSize(15).width(130);
              TextInput({
                text : this.wifiDeviceConfigManager?.staticIp?.gateway?.toString() ?? '' ,
                placeholder : "3232235777"
              })
                .fontSize("15vp")
                .onChange((strInput: string) => {
                  //判断合法性
                  if ( strInput.length >= 1 ) {
                    this.wifiDeviceConfigManager!.staticIp!.gateway = Number(strInput);
                  }
                })
                .width(ConfigData.WH_80_100)
                .borderRadius(1)
            }
            .backgroundColor($r("app.color.moon"))
            .padding(5)
            .justifyContent(FlexAlign.Start)
            .alignItems(VerticalAlign.Center)

            Row() {
              Text("staticIp_dnsServersM:").fontSize(15).width(140);
              TextInput({
                text : this.wifiDeviceConfigManager?.staticIp?.dnsServers?.join(', ') ?? '' ,
                placeholder : "3716386629"
              })
                .fontSize("15vp")
                .onChange((strInput: string) => {
                  const dnsServersArray = strInput.split(',').map(s => parseInt(s.trim() , 10)).filter(n => !isNaN(n));
                  //判断合法性
                  if ( dnsServersArray.length ) {
                    this.wifiDeviceConfigManager!.staticIp!.dnsServers = dnsServersArray;
                  } else {
                    // 处理输入不包含有效数字的情况
                    console.error('Invalid input for DNS servers');
                  }
                })
                .width(ConfigData.WH_80_100)
                .borderRadius(1)
            }
            .backgroundColor($r("app.color.moon"))
            .padding(5)
            .justifyContent(FlexAlign.Start)
            .alignItems(VerticalAlign.Center)

            Row() {
              Text("staticIp_domainsM:").fontSize(15).width(125);
              TextInput({
                text : this.wifiDeviceConfigManager?.staticIp?.domains?.join(', ') ?? '' , // 将数组转换成以逗号分隔的字符串
                placeholder : "0, 1, 2"
              })
                .fontSize("15vp")
                .onChange((strInput: string) => {
                  // 将输入的字符串按逗号分隔转换回数组
                  if ( strInput.length >= 1 ) {
                    this.wifiDeviceConfigManager!.staticIp!.domains = str