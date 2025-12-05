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
import router from '@ohos.router'

@Entry
@Component
struct router_PageA {
  @State amount: string = '0'
  @State page: number = 0

  aboutToAppear() {
    this.amount = router.getLength()
    this.page = router.getState().index;
  }

  build() {
    Scroll() {
      Column({ space: 5 }) {
        Text(this.amount).id('text_amount').fontSize(20).margin({ top: 50 })
        Text(this.page + '').id('text_page').fontSize(20).margin({ top: 50 })
        Button("跳转pageB")
          .type(ButtonType.Capsule)
          .margin({ bottom: 50 })
          .id("RouterA1")
          .onClick(() => {
            router.pushUrl({
              url: "MainAbility/pages/back_router/pageB",
            })
          })
        Button("返回上一级")
          .type(ButtonType.Capsule)
          .id("RouterA2")
          .onClick(() => {
            router.back()
          })
      }.width('100%')
    }
  }
}