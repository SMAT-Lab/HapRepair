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
struct ReplaceDestination1 {
  @Provide('pageInfos') pageInfos: NavPathStack = new NavPathStack()
  @State code100006: string = '';

  @Builder
  pageMap(name: string) {
    if (name == 'pageOne') {
      PageOne()
    } else if (name == 'pageThree') {
      PageThree()
    } else if (name == 'pageFour') {
      PageFour()
    }
  }

  build() {
    Navigation(this.pageInfos) {
      Column({space: '5vp'}) {
        Text(this.code100006).id('id06').height('5%')
        Button('push pageTwo').id('pageOneId').onClick(() => {
          this.pageInfos.replaceDestination({name: 'pageTwo'})
            .then(() => {
              console.log('[replaceDestination pageOne] success')
            })
            .catch((error: BusinessError) => {
              this.code100006 = error.code + '';
              console.log(`[replaceDestination pageOne] failed, error code ${error.code}`)
            })
        }).height('10%')

        Button('push undefined').id('undefinedId').onClick(() => {
          this.pageInfos.replaceDestination(undefined)
            .then(() => {
              console.log('[replaceDestination pageOne] success')
            })
            .catch((error: BusinessError) => {
              this.code100006 = error.code + '';
              console.log(`[replaceDestination undefined] failed, error code ${error.code}`)
            })
        }).height('10%')

        Button('push null').id('nullId').onClick(() => {
          console.log(`[replaceDestination null`)
          this.pageInfos.replaceDestination(null)
            .then(() => {
              console.log('[replaceDestination pageOne] success')
            })
            .catch((error: BusinessError) => {
              this.code100006 = error.code + '';
              console.log(`[replaceDestination null] failed, error code ${error.code}`)
            })
        }).height('10%')

        Button('replace pageOne').id('replaceId').onClick(() => {
          this.pageInfos.replaceDestination({name: 'pageOne'} )
            .then(() => {
              console.log('[replaceDestination pageOne] success')
            })
            .catch((error: BusinessError) => {
              this.code100006 = error.code + '';
              console.log(`[replaceDestination pageOne] failed, error code ${error.code}`)
            })
        }).height('10%')

        Button('push pageOne').id('pushId').onClick(() => {
          this.pageInfos.pushPath({name: 'pageOne'} )
        }).height('10%')
      }
      .width('100%')
      .height('100%')
    }
    .title('Title')
    .titleMode(NavigationTitleMode.Mini)
    .navDestination(this.pageMap)
  }
}

@Component
struct PageOne {
  @Consume('pageInfos') pageInfos: NavPathStack;
  @State stacks: string[] = [];
  build() {
    NavDestination() {
      Text(this.stacks.toString()).id('stackId').height('5%')
      Button('push pageThree').id('pushThreeId').onClick(() => {
        this.pageInfos.pushPath({name: 'pageThree'} )
      }).height('10%')
      Button('clear').id('clearId').onClick(() => {
        this.pageInfos.clear()
      }).height('10%')
    }
    .title('pageOne')
    .onReady(() => {
      this.stacks = this.pageInfos.getAllPathName()
      console.log('this.pageInfos: ', this.stacks)
    })
  }
}

@Component
struct PageThree {
  @Consume('pageInfos') pageInfos: NavPathStack;

  build() {
    NavDestination() {
      Text('This is page three')
      Button('push pageFour').id('pushFourId').onClick(() => {
        this.pageInfos.pushPath({name: 'pageFour'}, {launchMode: LaunchMode.STANDARD} )
      }).height('10%')
    }
    .title('page three')
    .onReady(() => {
      console.log('this.pageInfos: ', this.pageInfos.getAllPathName())
    })
  }
}

@Component
struct PageFour {
  @Consume('pageInfos') pageInfos: NavPathStack;

  build() {
    NavDestination() {
      Column({space: '5vp'}) {
        Button('LaunchModeMoveTOTopSingleton').id('moveId').onClick(() => {
          this.pageInfos.replaceDestination({ name: 'pageOne' }, { launchMode: LaunchMode.MOVE_TO_TOP_SINGLETON, animated: true })
        }).height('10%')
        Button('LaunchModePopToSingleton').id('popId').onClick(() => {
          this.pageInfos.replaceDestination({ name: 'pageOne' }, { launchMode: LaunchMode.POP_TO_SINGLETON, animated: undefined })
        }).height('10%')
      }
    }
    .title('PageFour')
    .onReady(() => {
      console.log('this.pageInfos: ', this.pageInfos.getAllPathName())
    })
  }
}