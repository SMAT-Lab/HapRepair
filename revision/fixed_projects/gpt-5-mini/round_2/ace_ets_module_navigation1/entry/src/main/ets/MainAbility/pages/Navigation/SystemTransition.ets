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
struct SystemTransition {
  @Provide('pageInfos') pageInfos: NavPathStack = new NavPathStack()

  @Builder
  pageMap(name: string) {
    if (name == 'pageOne') {
      PageOne()
    } else if (name == 'pageTwo') {
      PageTwo()
    }
  }

  build() {
    Navigation(this.pageInfos) {
      Column({space: '5vp'}) {
        Button('push pageOne').id('pageOneId').onClick(() => {
          this.pageInfos.pushPath({name: 'pageOne'})
        })
        Button('push pageTwo').id('pageTwoId').onClick(() => {
          this.pageInfos.pushPath({name: 'pageTwo'})
        })

      }
      .width('100%')
      .height('100%')
    }
    .title('Title')
    .navDestination(this.pageMap)
  }
}

@Component
struct PageOne {
  @Consume('pageInfos') pageInfos: NavPathStack;

  build() {
    NavDestination() {
      Text('This is page one').id('pageOne')
      Button('pop').id('popId').onClick(() => {
        this.pageInfos.pop()
      })
    }
    .title('pageOne')
    .systemTransition(NavigationSystemTransitionType.TITLE)
  }
}

@Component
struct PageTwo {
  @Consume('pageInfos') pageInfos: NavPathStack;

  build() {
    NavDestination() {
      Text('This is page two').id('pageTwo')
    }
    .title('pageTwo')
    .systemTransition(NavigationSystemTransitionType.CONTENT)
  }
}
