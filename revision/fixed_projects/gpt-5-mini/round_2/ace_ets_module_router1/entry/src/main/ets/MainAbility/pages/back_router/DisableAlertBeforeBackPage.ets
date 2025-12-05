/**
 * Copyright (c) 2025 Huawei Device Co., Ltd.
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

import router from '@system.router'

@Entry
@Component
struct DisableAlertBeforeBackPage {
  aboutToAppear() {
    router.disableAlertBeforeBackPage({
      success: () => {
        AppStorage.setOrCreate('disableSuccess', 1);
      },
      cancel: () => {
        console.log('AlertBefore fail')
      },
      complete: () => {
        AppStorage.setOrCreate('disableComplete', 1);
      }
    })
  }

  build() {
    Flex({ direction: FlexDirection.Column, alignItems: ItemAlign.Center, justifyContent: FlexAlign.Center }) {
      Button('disableAlertBeforeBackPage')
        .width(100)
        .height('10%')
        .id('disableId')
        .onClick(() => {
          router.back({uri:'MainAbility/pages/back_router/RouterBackAlert', params: '123'});
        });
    }.width('100%').height('100%')
  }
}