
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

interface RouterParams {
  view: string;
}

@Entry
@Component
struct RouterPushTwo {
  aboutToAppear() {
    hilog.info(0x0000, 'testTag', '%{public}s', 'TestAbility index aboutToAppear');
  }

  onPageShow() {
    let uiContext: UIContext = this.getUIContext?.()
    let routerUI: Router = uiContext.getRouter();
    this.routerUI = routerUI
  }
  // 直接使用一般变量即可
  params: string = router.getParams() ? (router.getParams() as RouterParams).view : ""
  @State routerUI: Router|undefined = undefined

  build() {
    Scroll() {
      Column() {
        Text("RouterPushTwo " + this.params)
          .fontSize(50)
          .fontWeight(FontWeight.Bold)
          .id("Text_RouterPushTwo")

        Button("turn_RouterPushUrlOne_RouterMode_Standard")
          .type(ButtonType.Capsule)
          .margin({
            top: 20
          })
          .id("turn_RouterPushUrlOne_RouterMode_Standard")
          .onClick(() => {
            router.pushUrl({
              url: 'MainAbility/pages/State/RouterPushOne',
            }, router.RouterMode.Standard)
          })

        Button("turn_RouterPushUrlOne_RouterMode_Single")
          .type(ButtonType.Capsule)
          .margin({
            top: 20
          })
          .id("turn_RouterPushUrlOne_RouterMode_Single")
          .onClick(() => {
            router.pushUrl({
              url: 'MainAbility/pages/State/RouterPushOne',
            }, router.RouterMode.Single)
          })

        Text("UIRouterPushTwo " + this.params)
          .fontSize(50)
          .fontWeight(FontWeight.Bold)
          .id("Text_UIRouterPushTwo")

        Button("turn_UIRouterPushUrlOne_RouterMode_Standard")
          .type(ButtonType.Capsule)
          .margin({
            top: 20
          })
          .id("turn_UIRouterPushUrlOne_RouterMode_Standard")
          .onClick(() => {
            this.routerUI?.pushUrl({
              url: 'MainAbility/pages/State/RouterPushOne',
            }, router.RouterMode.Standard)
          })

        Button("turn_UIRouterPushUrlOne_RouterMode_Single")
          .type(ButtonType.Capsule)
          .margin({
            top: 20
          })
          .id("turn_UIRouterPushUrlOne_RouterMode_Single")
          .onClick(() => {
            this.routerUI?.pushUrl({
              url: 'MainAbility/pages/State/RouterPushOne',
            }, router.RouterMode.Single)
          })
      }
    }
  }
}