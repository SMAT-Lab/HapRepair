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
struct pageB {
  build() {
    Row() {
      Column() {
        Text('我是B')
          .fontSize(20)
          .fontWeight(FontWeight.Bold)

        Button('清空所有栈').onClick(() => {
          router.clear()
        }).margin({ bottom: 20 }).id('clear_B')

        Button('更新到C页面').onClick(() => {
          router.pushUrl({
            url: 'MainAbility/pages/Router2/pageC',
          });
        }).margin({ top: 20 }).id('page_c')

        Text('' + router.getState()['name']).id('name_B')
        Text('' + router.getLength()).id('name_length')

      }
      .width('100%')
    }
    .height('100%')
  }
}