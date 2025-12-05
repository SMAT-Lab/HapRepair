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

import { uiObserver } from '@kit.ArkUI';

@Component
struct UniqueIdPageTwo {
  @Consume('stack') stack:NavPathStack;

  private text = '';
  private uniqueId  = -1;

  aboutToAppear() {
    let navigationUniqueId = this.queryNavigationInfo()?.uniqueId;
    if(navigationUniqueId) {
      this.uniqueId = navigationUniqueId.valueOf();
    }
    this.text = JSON.stringify(this.uniqueId);

    let navigationInfo: uiObserver.NavigationInfo | undefined = this.queryNavigationInfo()
    console.log('queryNavigationInfo: ' + JSON.stringify(navigationInfo))
  }

  build() {
    NavDestination() {
      Text('pageTwo')
        .height('5%')
      Text('navigationUniqueId:' + this.text)
        .height('5%')
      Button('back').onClick(() => {
        this.stack.pop();
      })
        .height('5%')
        .id('backId')
    }.title('pageTwo')
  }
}

@Component
struct UniqueIdPageOne {
  @Consume('stack') stack:NavPathStack;

  private text = '';
  private uniqueId  = -1;

  info1 = (info: uiObserver.NavDestinationInfo) => {
    if (info && info.state == 8) {
      console.info('NavDestination state update navigationId', JSON.stringify(info));
      AppStorage.setOrCreate('navDestinationName', info.name);
    }
  }

  aboutToAppear() {
    let navigationUniqueId = this.queryNavigationInfo()?.uniqueId;
    if(navigationUniqueId) {
      this.uniqueId = navigationUniqueId.valueOf();
    }
    this.text = JSON.stringify(this.uniqueId);
    AppStorage.setOrCreate('uniqueId', this.text);
  }

  build() {
    NavDestination() {
      Text('pageOne')
        .height('5%')
      Text('navigationUniqueId:' + this.text)
        .height('5%')
      Button('uiObserver.on')
        .onClick(() => {
          this.getUIContext().getUIObserver().on('navDestinationUpdateByUniqueId', this.uniqueId, this.info1);
        })
        .id('onId')
      Button('uiObserver.off')
        .onClick(() => {
          uiObserver.off('navDestinationUpdate');
          uiObserver.off('navDestinationUpdate', { navigationId: 'testId' });
          this.getUIContext().getUIObserver().off('navDestinationUpdateByUniqueId', this.uniqueId, this.info1);
          console.info('close getNavigationInfoByUniqueId: ', this.uniqueId);
        })
        .id('offId')
      Button('push pageTwo').onClick(() => {
        this.stack.pushPath({ name: 'pageTwo' });
      })
        .id('pushTwoId')
      Button('back').onClick(() => {
        this.stack.pop();
      })
        .height('5%')
        .id('backId')
    }.title('pageOne')
  }
}

@Entry
@Component
struct UIContextNavDestinationUpdate {
  @Provide stack: NavPathStack = new NavPathStack();

  @Builder
  PageBuilder(name: string) {
    if (name == 'pageOne') {
      UniqueIdPageOne()
    } else if (name == 'pageTwo') {
      UniqueIdPageTwo()
    }
  }

  build() {
    Column() {
      Navigation(this.stack) {
        Button('Push PageOne')
          .onClick(() => {
            this.stack.pushPath({ name: 'pageOne' })
          })
          .id('pushOneId')
      }
      .id('testId')
      .title('Navigation')
      .navDestination(this.PageBuilder)
      .titleMode(NavigationTitleMode.Mini)
    }
    .width('100%')
    .height('100%')
  }
}