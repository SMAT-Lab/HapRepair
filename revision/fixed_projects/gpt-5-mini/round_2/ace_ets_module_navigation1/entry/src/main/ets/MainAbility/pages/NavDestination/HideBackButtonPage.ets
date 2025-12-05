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

@Entry
@Component
struct HideBackButtonPage {
  @Provide('pageInfos') pageInfos: NavPathStack = new NavPathStack()

  @Builder
  pageMap(name: string) {
    if (name == 'pageOne') {
      PageOne27()
    } else if (name == 'pageTwo') {
      PageTwo27()
    } else if (name == 'pageThree') {
      PageThree2()
    } else if (name == 'pageFour') {
      PageFour()
    }
  }

  build() {
    Navigation(this.pageInfos) {
      Column({space: '5vp'}) {
        Button('push pageOne').id('pushOneId').onClick(() => {
          this.pageInfos.pushPath({name: 'pageOne'} )
        }).height('10%')

        Button('push pageTwo').id('pushTwoId').onClick(() => {
          this.pageInfos.pushPath({name: 'pageTwo'} )
        }).height('10%')

        Button('push pageThree').id('pageThreeId').onClick(() => {
          this.pageInfos.pushPath({name: 'pageThree'} )
        }).height('10%')

        Button('push pageFour').id('pushFourId').onClick(() => {
          this.pageInfos.pushPath({name: 'pageFour'} )
        }).height('10%')
      }
      .width('100%')
      .height('100%')
    }
    .title('Title')
    .titleMode(NavigationTitleMode.Mini)
    .navDestination(this.pageMap)
    .mode(NavigationMode.Stack)
  }
}

@Component
struct PageOne27 {
  @Consume('pageInfos') pageInfos: NavPathStack;
  build() {
    NavDestination() {
      Button('pop').id('popId').onClick(() => {
        this.pageInfos.pop()
      }).height('10%')
    }
    .title('pageOne')
    .hideBackButton(true)
  }
}

@Component
struct PageTwo27 {
  @Consume('pageInfos') pageInfos: NavPathStack;
  build() {
    NavDestination() {
      Button('pop').id('popId').onClick(() => {
        this.pageInfos.pop()
      }).height('10%')
    }
    .title('pageTwo')
    .hideBackButton(false)
  }
}

@Component
struct PageThree2 {
  @Consume('pageInfos') pageInfos: NavPathStack;
  build() {
    NavDestination() {
      Button('pop').id('popId').onClick(() => {
        this.pageInfos.pop()
      }).height('10%')
    }
    .title('pageThree')
    .hideBackButton(null)
  }
}

@Component
struct PageFour {
  @Consume('pageInfos') pageInfos: NavPathStack;
  build() {
    NavDestination() {
      Button('pop').id('popId').onClick(() => {
        this.pageInfos.pop()
      }).height('10%')
    }
    .title('pageThree')
    .hideBackButton(undefined)
  }
}