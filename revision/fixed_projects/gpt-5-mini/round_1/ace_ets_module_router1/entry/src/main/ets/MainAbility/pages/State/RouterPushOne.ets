
/**
 * Copyright (c) 2023 Huawei Device Co., Ltd.
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

import hilog from '@ohos.hilog';
import router from '@ohos.router';
import { Router, UIContext } from '@ohos.arkui.UIContext';

@Entry
@Component
struct RouterPushOne {
  aboutToAppear() {
    hilog.info(0x0000, 'testTag', '%{public}s', 'TestAbility index aboutToAppear');

  }

  onPageShow() {
    let uiContext: UIContext = this.getUIContext?.()
    let routerUI: Router = uiContext.getRouter();
    this.routerUI = routerUI
  }

  // 直接使用一般变量即可
  message: string = 'RouterPushUrlOne'
  messageUI: string = 'UIRouterPushUrlOne'
  @State routerUI: Router|undefined = undefined

  build() {
    Scroll() {
      Column() {
        Text(this.message).height('5%')
        Text("RouterPushThree " + JSON.stringify(router.getParams()))
          .fontSize(20)
          .id("Text_RouterPushThree_Params").height('5%')
        Button("turn_RouterPushUrlTwo_noParms")
          .type(ButtonType.Capsule)
          .margin({
            top: 20
          })
          .id("turn_RouterPushUrlTwo_noParms")
          .onClick(() => {
            router.pushUrl({
              url: 'MainAbility/pages/State/RouterPushTwo'
            })
          }).height('5%')
        Button("turn_RouterPushUrlTwo_hasParms")
          .type(ButtonType.Capsule)
          .margin({
            top: 20
          })
          .id("turn_RouterPushUrlTwo_hasParms")
          .onClick(() => {
            router.pushUrl({
              url: 'MainAbility/pages/State/RouterPushTwo',
              params: {
                view: "params"
              }
            })
          }).height('5%')

        Button("turn_RouterPushReplace_noParms")
          .type(ButtonType.Capsule)
          .margin({
            top: 20
          })
          .id("turn_RouterPushReplace_noParms")
          .onClick(() => {
            router.replaceUrl({
              url: 'MainAbility/pages/State/RouterPushTwo'
            })
          }).height('5%')
        Button("turn_RouterPushReplace_hasParms")
          .type(ButtonType.Capsule)
          .margin({
            top: 20
          })
          .id("turn_RouterPushReplace_hasParms")
          .onClick(() => {
            router.replaceUrl({
              url: 'MainAbility/pages/State/RouterPushTwo',
              params: {
                view: "params"
              }
            })
          }).height('5%')

        Text(this.messageUI).height('5%')
        Button("turn_UIRouterPushUrlTwo_noParms")
          .type(ButtonType.Capsule)
          .margin({
            top: 20
          })
          .id("turn_UIRouterPushUrlTwo_noParms")
          .onClick(() => {
            this.routerUI?.pushUrl({
              url: 'MainAbility/pages/State/RouterPushTwo'
            })
          }).height('5%')
        Button("turn_UIRouterPushUrlTwo_hasParms")
          .type(ButtonType.Capsule)
          .margin({
            top: 20
          })
          .id("turn_UIRouterPushUrlTwo_hasParms")
          .onClick(() => {
            this.routerUI?.pushUrl({
              url: 'MainAbility/pages/State/RouterPushTwo',
              params: {
                view: "params"
              }
            })
          }).height('5%')

        Button("turn_UIRouterPushReplace_noParms")
          .type(ButtonType.Capsule)
          .margin({
            top: 20
          })
          .id("turn_UIRouterPushReplace_noParms")
          .onClick(() => {
            this.routerUI?.replaceUrl({
              url: 'MainAbility/pages/State/RouterPushTwo'
            })
          }).height('5%')
        Button("turn_UIRouterPushReplace_hasParms")
          .type(ButtonType.Capsule)
          .margin({
            top: 20
          })
          .id("turn_UIRouterPushReplace_hasParms")
          .onClick(() => {
            this.routerUI?.replaceUrl({
              url: 'MainAbility/pages/State/RouterPushTwo',
              params: {
                view: "params"
              }
            })
          }).height('5%')
      }
    }
  }
}