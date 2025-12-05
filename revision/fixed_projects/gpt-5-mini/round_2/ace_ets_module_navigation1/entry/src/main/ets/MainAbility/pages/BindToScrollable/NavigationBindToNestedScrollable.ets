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

import { SymbolGlyphModifier } from '@kit.ArkUI';

@Entry
@Component
struct NavigationBindToNestedScrollable {
  @Provide('navPathStack') navPathStack: NavPathStack = new NavPathStack();

  @Builder
  myRouter(name:string, param?:Object) {
    if (name === 'PageOne') {
      PageOne();
    }
  }

  build() {
    Navigation(this.navPathStack) {
      Column() {
        Button('push pageOne').onClick(()=>{
          this.navPathStack.pushPathByName('PageOne', null)
        })
      }
    }
    .titleMode(NavigationTitleMode.Mini)
    .title('一级页面')
    .navDestination(this.myRouter)
  }
}

@Component
export struct PageOne {
  @Consume('navPathStack') navPathStack:NavPathStack;
  private arr: number[] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
  private arr1: number[] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  scroller: Scroller = new Scroller()
  scroller1: Scroller = new Scroller()

  build() {
    NavDestination() {
      Row() {
        Scroll(this.scroller) {
          Column({space: 10}) {
            ForEach(this.arr, (item: number) => {
              if (item == 0) {
                List({space: 10, scroller: this.scroller1}) {
                  ForEach(this.arr1, (item1: number) => {
                    ListItem() {
                      Text('Child_item_' + item1).fontSize(16).backgroundColor(Color.Pink).width('90%').height(100)
                    }
                  }, (item1: string) => item1)
                }
                .height('70%')
                .margin({right: '60%'})
              } else {
                Text('item_' + item).fontSize(16).backgroundColor(Color.Yellow).width('90%').height(100)
              }
            }, (item: string) => item)
          }
        }
        .scrollable(ScrollDirection.Vertical)
        .scrollBar(BarState.On)
        .scrollBarColor(Color.Gray)
        .scrollBarWidth(10)
        .friction(0.6)
        .edgeEffect(EdgeEffect.Spring)
      }
      .height('100%')
    }
    .title('pageOne', {backgroundColor: Color.Red})
    .bindToNestedScrollable([{parent: this.scroller, child: this.scroller1}])
    .toolbarConfiguration([
      {value: 'a', symbolIcon: new SymbolGlyphModifier($r('sys.symbol.ohos_folder_badge_plus'))},
      {value: 'b', symbolIcon: new SymbolGlyphModifier($r('sys.symbol.ohos_folder_badge_plus'))},
      {value: 'c', symbolIcon: new SymbolGlyphModifier($r('sys.symbol.ohos_folder_badge_plus'))},
      {value: 'd', symbolIcon: new SymbolGlyphModifier($r('sys.symbol.ohos_folder_badge_plus'))},
      {value: 'e', symbolIcon: new SymbolGlyphModifier($r('sys.symbol.ohos_folder_badge_plus'))},
    ])
    .id('navId')
    .margin({top: '5%'})
  }
}