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
struct ReplaceDestination2 {
  @Provide('pageInfos') pageInfos: NavPathStack = new NavPathStack()
  @State code100005: string = '';

  @Builder
  pageMap(name: string) {
    if (name == 'pageOne') {
      PageOne()
    }
  }

  build() {
    Navigation(this.pageInfos) {
      Column({space: '5vp'}) {
        Text(this.code100005).id('id05')
        Button('push pageOne').id('pageTwoId').onClick(() => {
          this.pageInfos.replaceDestination({name: 'pageOne'})
            .then(() => {
              console.log('[replaceDestination pageOne] success')
            })
            .catch((error: BusinessError) => {
              this.code100005 = error.code + '';
              console.log(`[replaceDestination pageOne] failed, error code ${error.code}`)
            })
        })
      }
      .width('100%')
      .height('100%')
    }
    .title('Title')
  }
}

@Component
struct PageOne {
  @Consume('pageInfos') pageInfos: NavPathStack;

  build() {
    NavDestination() {
      Text('This is page one')
    }
    .title('pageOne')
  }
}