/**
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
// PageTwo.ets
export class Pages {
  names: string = ""
  values: NavPathStack | null = null
}

class resultClass {
  constructor(count: number) {
    this.count = count;
  }
  count: number = 10
}

@Builder
export function pageTwoTmp(info: Pages) {
  NavDestination() {
    Column() {
      Text('' + info)
      Text('two').id('NavigationTwo')
      Button('pop', { stateEffect: true, type: ButtonType.Capsule })
        .width('80%')
        .height(40)
        .margin(20)
        .onClick(() => {
          (info.values as NavPathStack).pop(new resultClass(1)); // 回退到上一个页面，将处理结果传入push的onPop回调中。
        })

      Button('popToName', { stateEffect: true, type: ButtonType.Capsule })
        .width('80%')
        .height(40)
        .margin(20)
        .onClick(() => {
          (info.values as NavPathStack).popToName('pageOne', new resultClass(11)); // 将第一个名为name的NavDestination页面移到栈顶，将处理结果传入push的onPop回调中。
        })

      Button('popToIndex', { stateEffect: true, type: ButtonType.Capsule })
        .width('80%')
        .height(40)
        .margin(20)
        .onClick(() => {
          (info.values as NavPathStack).popToIndex(0, new resultClass(111)); // 将index指定的NavDestination页面移到栈顶，将处理结果传入push的onPop回调中。
        })

      Button('popWithoutResult', { stateEffect: true, type: ButtonType.Capsule })
        .width('80%')
        .height(40)
        .margin(20)
        .onClick(() => {
          (info.values as NavPathStack).pop();
        })

      Button('popToNameWithoutResult', { stateEffect: true, type: ButtonType.Capsule })
        .width('80%')
        .height(40)
        .margin(20)
        .onClick(() => {
          (info.values as NavPathStack).popToName('pageOne');
        })

      Button('popToIndexWithoutResult', { stateEffect: true, type: ButtonType.Capsule })
        .width('80%')
        .height(40)
        .margin(20)
        .onClick(() => {
          (info.values as NavPathStack).popToIndex(0);
        })
    }.width('100%').height('100%')
  }.title('pageTwo')
  .onBackPressed(() => {
    (info.values as NavPathStack).pop(new resultClass(0)); // 回退到上一个页面，将处理结果传入push的onPop回调。
    return true;
  })
}