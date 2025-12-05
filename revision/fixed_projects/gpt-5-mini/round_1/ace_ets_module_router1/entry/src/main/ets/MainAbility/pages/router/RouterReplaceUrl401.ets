/**
 * Copyright (c) 2025 Shenzhen Kaihong Digital Industry Development Co., Ltd.
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

@Entry
@Component
struct RouterReplaceUrl401 {
  @State errCode: string = '';

  build() {
    Flex({ justifyContent: FlexAlign.Center, alignItems: ItemAlign.Center }) {
      Column() {
        Text(this.errCode).id('id06').height('10%')
        Button('routerCallbackPush401')
          .height('10%')
          .id('401Btn2')
          .onClick(() => {
            try {
              router.replaceUrl({
                url: undefined,
                params: 'name',
                recoverable: true,
              }, (err) => {
                if (err) {
                  console.error(`replaceUrl failed, code is ${err.code}, message is ${err.message}`);
                  return;
                }
              })
            } catch (err) {
              this.errCode = err.code + '';
              console.error(`replaceUrl failed, code is ${err.code}, message is ${err.message}`)
              AppStorage.setOrCreate('401Code', err.code);
            }
          })
      }
    }
  }
}