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
struct PageOne {
  @Consume('stack') stack:NavPathStack;
  build() {
    NavDestination() {
      Text('pageOne')
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
struct NavDestinationStateActive {
  @Provide stack: NavPathStack = new NavPathStack();

  @Builder
  PageBuilder(name: string) {
    PageOne()
  }

  aboutToAppear() {
    uiObserver.on('navDestinationUpdate', (info) => {
      console.info('NavDestination state update', JSON.stringify(info));
      if (info.state == uiObserver.NavDestinationState.ON_ACTIVE) {
        AppStorage.setOrCreate('ON_ACTIVE', info.state);
      }
      if (info.state == uiObserver.NavDestinationState.ON_INACTIVE) {
        AppStorage.setOrCreate('ON_INACTIVE', info.state);
      }
    });
  }

  aboutToDisappear() {
    uiObserver.off('navDestinationUpdate');
  }

  build() {
    Column() {
      Navigation(this.stack) {
        Button('push').onClick(() => {
          this.stack.pushPath({ name: 'pageOne' });
        })
        .height('5%')
        .id('pushId')
      }
      .title('Navigation')
      .navDestination(this.PageBuilder)
      .mode(NavigationMode.Stack)
    }
    .width('100%')
    .height('100%')
  }
}