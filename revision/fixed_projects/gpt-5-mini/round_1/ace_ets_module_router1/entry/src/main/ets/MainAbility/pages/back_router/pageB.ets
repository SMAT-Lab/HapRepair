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
struct router_PageB {
  @State amount: string = '0'
  @State page: number = 0

  aboutToAppear() {
    this.amount = router.getLength()
    this.page = router.getState().index;
  }

  build() {
    Scroll() {
      Column({ space: 5 }) {
        Text(this.amount).id('amount1').fontSize(20).margin({ top: 50 })
        Text(this.page + '').id('page1').fontSize(20).margin({ top: 50 })
        Button("跳转pageC")
          .type(ButtonType.Capsule)
          .margin({ bottom: 50 })
          .id("RouterB1")
          .onClick(() => {
            router.pushUrl({
              url: "MainAbility/pages/back_router/pageC"
            })
          })
        Button("返回上一级")
          .type(ButtonType.Capsule)
          .id("RouteB2")
          .onClick(() => {
            router.back()
          })
        Button('CLEAR')
          .id('clear')
          .type(ButtonType.Capsule)
          .margin(20)
          .onClick(() => {
            router.clear()
            this.amount = router.getLength()
            this.page = router.getState().index;
          })
        Button("ReturnBack")
          .type(ButtonType.Capsule)
          .id("ReturnBack")
          .onClick(() => {
            router.back({ url: 'pages/back_router/pageB' });
            this.amount = router.getLength()
            this.page = router.getState().index;
          })
        Text(this.amount).id('index')
        Text(this.page + '').id('length')
      }.width('100%')
    }
  }
}