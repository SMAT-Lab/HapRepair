/*
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

import { BusinessError } from '@kit.BasicServicesKit';

@Entry
@Component
struct PushDestination401 {
  @Provide('pageInfos') pageInfos: NavPathStack = new NavPathStack()
  @State errCode: string = '';

  build() {
    Navigation(this.pageInfos) {
      Column({space: '5vp'}) {
        Text(this.errCode).id('id06').height('5%')
        Button('push undefined').id('undefinedId').onClick(() => {
          this.pageInfos.pushDestination(undefined, false)
            .then(() => {
              console.log('[pushDestination pageOne] success')
            })
            .catch((error: BusinessError) => {
              this.errCode = error.code + '';
              console.log(`[pushDestination undefined] failed, error code ${error.code}`)
            })
        }).height('10%')
        Button('push null').id('nullId').onClick(() => {
          this.pageInfos.pushDestination(null, { launchMode: LaunchMode.STANDARD, animated: true })
            .then(() => {
              console.log('[pushDestination pageOne] success')
            })
            .catch((error: BusinessError) => {
              this.errCode = error.code + '';
              console.log(`[pushDestination null] failed, error code ${error.code}`)
            })
        }).height('10%')
      }
      .width('100%')
      .height('100%')
    }
    .title('Title')
    .titleMode(NavigationTitleMode.Mini)
  }
}