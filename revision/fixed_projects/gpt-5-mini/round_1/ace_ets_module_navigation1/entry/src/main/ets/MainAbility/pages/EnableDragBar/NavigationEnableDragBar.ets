
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
struct NavigationEnableDragBar {
  @Provide('pageInfos') pageInfos: NavPathStack = new NavPathStack()
  private arr: number[] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  currentIndex: number = 0;
  @State enableDrag: Optional<boolean> = false;

  @Builder
  NavigationTitle() {
    Column() {
      Text('Title')
        .fontColor('#182431')
        .fontSize(30)
        .lineHeight(41)
        .fontWeight(700)
      Text('subtitle')
        .fontColor('#182431')
        .fontSize(14)
        .lineHeight(19)
        .opacity(0.4)
        .margin({ top: 2, bottom: 20 })
    }.alignItems(HorizontalAlign.Start)
  }

  @Builder
  NavigationMenus() {
    Row() {
      Image('resources/base/media/ic_public_add.svg')
        .width(24)
        .height(24)
      Image('resources/base/media/ic_public_add.svg')
        .width(24)
        .height(24)
        .margin({ left: 24 })
      Image('common/ic_public_more.svg')
        .width(24)
        .height(24)
        .margin({ left: 24 })
    }
  }

  build() {
    Navigation(this.pageInfos) {
      Column() {
        Scroll() {
          Column() {
              Button('拖拽条开启').id('drag_true')
                .onClick(() => {
                  this.enableDrag = true
                })
              Button('拖拽条关闭').id('drag_false')
                .onClick(() => {
                  this.enableDrag = false
                })
              Button('拖拽条undefined').id('drag_undefined')
                .onClick(() => {
                  this.enableDrag = undefined
                })
            }
          }
        }
      }
    }
    .enableDragBar(this.enableDrag)
    .id('navigatinId')
    .navBarPosition(NavBarPosition.Start)
    .navBarWidthRange([100, 500])
    .titleMode(NavigationTitleMode.Mini)
    .navBarWidth(250)
    .minContentWidth(100)
    .title(this.NavigationTitle)
    .menus(this.NavigationMenus)
    .hideTitleBar(false)
    .hideToolBar(false)
    .onTitleModeChange((titleModel: NavigationTitleMode) => {
      console.info('titleMode' + titleModel)
    })
    .width('100%')
    .height('100%')
    .backgroundColor('#F1F3F5')
  }
}