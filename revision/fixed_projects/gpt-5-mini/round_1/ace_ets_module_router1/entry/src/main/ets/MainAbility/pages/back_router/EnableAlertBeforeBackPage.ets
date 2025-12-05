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
struct EnableAlertBeforeBackPage {
  aboutToAppear() {
    router.enableAlertBeforeBackPage({
      message: 'Message Info',
      success: () => {
        AppStorage.setOrCreate('enableSuccess', 1);
      },
      cancel: () => {
        AppStorage.setOrCreate('enableCancel', 1);
      },
      complete: () => {
        AppStorage.setOrCreate('enableComplete', 1);
      }
    })
  }

  build() {
    Flex({ direction: FlexDirection.Column, alignItems: ItemAlign.Center, justifyContent: FlexAlign.Center }) {
      Button('enableAlertBeforeBackPage')
        .width(100)
        .height('10%')
        .id('enableId')
        .onClick(() => {
          router.back();
        });
    }.width('100%').height('100%');
  }
}