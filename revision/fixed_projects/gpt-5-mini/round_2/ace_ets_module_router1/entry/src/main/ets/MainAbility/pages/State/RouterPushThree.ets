/**
 * Copyright (c) 2024 Huawei Device Co., Ltd.
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

import router from '@ohos.router';
import { Router, UIContext } from '@ohos.arkui.UIContext';


@Entry
@Component
struct RouterPushThree {
  build() {
    Scroll() {
      Column() {
        Text("RouterThree")
          .fontSize(20)
          .fontWeight(FontWeight.Bold)

        Button("turn_RouterPushUrlOne_null")
          .type(ButtonType.Capsule)
          .margin({
            top: 10
          })
          .id("turn_RouterPushUrlOne_null")
          .onClick(() => {
            router.replaceUrl({
              url: 'MainAbility/pages/State/RouterPushOne',
              params: null
            })
          })

        Button("turn_RouterPushUrlOne_string_number")
          .type(ButtonType.Capsule)
          .margin({
            top: 10
          })
          .id("turn_RouterPushUrlOne_string_number")
          .onClick(() => {
            router.replaceUrl({
              url: 'MainAbility/pages/State/RouterPushOne',
              params: {
                data1: {
                  data2: "message",
                  data3: 12345
                }
              }
            })
          })

        Button("turn_RouterPushThree_stringArray")
          .type(ButtonType.Capsule)
          .margin({
            top: 10
          })
          .id("turn_RouterPushThree_stringArray")
          .onClick(() => {
            router.replaceUrl({
              url: 'MainAbility/pages/State/RouterPushOne',
              params: {
                data1: {
                  data2: ["message", "message1"]
                }
              }
            })
          })

        Button("turn_RouterPushUrlOne_string_number1")
          .type(ButtonType.Capsule)
          .margin({
            top: 10
          })
          .id("turn_RouterPushUrlOne_string_number1")
          .onClick(() => {
            router.replaceUrl({
              url: 'MainAbility/pages/State/RouterPushOne',
              params: {
                data1: "message",
                data2: 12345
              }
            })
          })

        Button("turn_RouterPushUrlOne_string_numArray")
          .type(ButtonType.Capsule)
          .margin({
            top: 10
          })
          .id("turn_RouterPushUrlOne_string_numArray")
          .onClick(() => {
            router.replaceUrl({
              url: 'MainAbility/pages/State/RouterPushOne',
              params: {
                data1: "message",
                data2: [1, 2, 3, 4, 5]
              }
            })
          })

        Button("turn_RouterPushUrlOne_string_number_push")
          .type(ButtonType.Capsule)
          .margin({
            top: 10
          })
          .id("turn_RouterPushUrlOne_string_number_push")
          .onClick(() => {
            router.pushUrl({
              url: 'MainAbility/pages/State/RouterPushOne',
              params: {
                data1: {
                  data2: "message",
                  data3: 12345
                }
              }
            })
          })

        Button("turn_RouterPushUrlOne_null_push")
          .type(ButtonType.Capsule)
          .margin({
            top: 10
          })
          .id("turn_RouterPushUrlOne_null_push")
          .onClick(() => {
            router.pushUrl({
              url: 'MainAbility/pages/State/RouterPushOne',
              params: null
            })
          })
      }
    }
  }
}
