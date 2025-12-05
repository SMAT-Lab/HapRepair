/**
 * Copyright (c) 2022 Shenzhen Kaihong Digital Industry Development Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import router from '@ohos.router'
import { BusinessError } from '@ohos.base';
// import uiAppearance from '@ohos.uiAppearance';

import Log from '../../common/Log';

const TAG = 'ets_apiLack_add';

@Entry
@Component
struct RouterStandard {
  async routePage() {
    let options: router.RouterOptions = {
      url: 'pages/ohosRouter/page1',
      params: {
        text: '这是第一页的值',
        data: [12, 45, 78]
      }
    }
    try {
      await router.pushUrl(options)
    } catch (err) {
      console.info(`Standard Page fail callback, code: ${(err as BusinessError).code}, msg: ${(err as BusinessError).message}`)
    }
  }

  async routeSinglePage() {
    let options: router.RouterOptions = {
      url: 'pages/ohosRouter/page2',
      params: {
        text: '这是第二页的值',
        data: [13, 46, 79]
      }
    }
    try {
      await router.pushUrl(options)
    } catch (err) {
      console.info(`Single Page fail callback, code: ${(err as BusinessError).code}, msg: ${(err as BusinessError).message}`)
    }
  }

  aboutToAppear() {
    Log.showInfo(TAG, `aboutToAppear RouterStandard start`)
  }

  aboutToDisappear() {
    Log.showInfo(TAG, `aboutToDisAppear RouterStandard end`)
  }

  build() {

    Flex({ direction: FlexDirection.Column, alignItems: ItemAlign.Center, justifyContent: FlexAlign.Center }) {

      Text("router-Standard")
        .width(100)
        .height(70)
        .fontSize(20)
        .opacity(1)
        .align(Alignment.TopStart)
        .fontColor(0xCCCCCC)
        .lineHeight(25)
        .border({ width: 1 })
        .padding(10)
        .textAlign(TextAlign.Center)
        .textOverflow({ overflow: TextOverflow.None })
        .key("StandardText")
        .onClick(() => {
          this.routePage()
        })

      Text("router-Single")
        .width(100)
        .height(70)
        .fontSize(20)
        .opacity(1)
        .align(Alignment.TopStart)
        .fontColor(0xCCCCCC)
        .lineHeight(25)
        .border({ width: 1 })
        .padding(10)
        .textAlign(TextAlign.Center)
        .textOverflow({ overflow: TextOverflow.None })
        .key("SingleText")
        .onClick(() => {
          this.routeSinglePage()
        })

      Text('这是第一页')
        .fontSize(50)
        .fontWeight(FontWeight.Bold)
      Button() {
        Text('next page')
          .fontSize(25)
          .fontWeight(FontWeight.Bold)
      }.type(ButtonType.Capsule)
      .margin({ top: 20 })
      .backgroundColor('#ccc')
      .onClick(() => {
        this.routePage()
      })
    }
    .width("100%")
    .height("100%")
  }
}

