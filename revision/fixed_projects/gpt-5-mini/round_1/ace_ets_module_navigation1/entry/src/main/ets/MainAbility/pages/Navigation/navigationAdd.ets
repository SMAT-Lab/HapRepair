/**
 * Copyright (c) 2023 Huawei Device Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the 'License');
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an 'AS IS' BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
import promptA from '@ohos.promptAction';
@Entry
@Component
struct NavigationExample2 {
  private arr: number[] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  currentIndex: number = 0
  navPathStack = new NavPathStack();
  private value:Object = [1,24,5,6]


  @Builder NavigationTitle() {
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

  @Builder NavigationMenus() {
    Row() {
      Text('1')
        .width(24)
        .height(24)
      Text('2')
        .width(24)
        .height(24)
        .margin({ left: 24 })
    }
  }


  build() {
    Stack({ alignContent: Alignment.TopStart }) {
      Navigation(this.navPathStack) {
        TextInput({ placeholder: 'search...' })
          .width('90%')
          .height(40)
          .enableKeyboardOnFocus(false)
          .backgroundColor('#FFFFFF')
          .margin({ top: 8 })
        List({ space: 12, initialIndex: 0 }) {
          ForEach(this.arr, (item: number) => {
            ListItem() {
              Text('' + item)
                .width('90%')
                .height(72)
                .backgroundColor('#FFFFFF')
                .borderRadius(24)
                .fontSize(16)
                .fontWeight(500)
                .textAlign(TextAlign.Center)
            }.editable(true)
          }, (item: string) => item)
        }
        .height(324)
        .width('100%')
        .margin({ top: 12, left: '10%' })
      }
      .title(this.NavigationTitle)
      .menus(this.NavigationMenus)
      .titleMode(NavigationTitleMode.Full)
      .hideTitleBar(false)
      .key('navigationAdd')
      .hideToolBar(false)
      .navDestination((name: string, parm: Object) => {
          this.navPathStack.pushPathByName(name,parm)
          promptA.showToast({message:name})

      })
      .onTitleModeChange((titleModel: NavigationTitleMode) => {
        console.info('titleMode' + titleModel)
      })

      Button('pushPathByName')
        .height('5%')
        .onClick(() => { // 点击后下滑指定距离150.0vp
          this.navPathStack.pushPathByName('navidesnation_add',this.value)
        })
        .margin({ top: 10, left: 20 })
      Button('popToName')
        .height('5%')
        .onClick(() => {
          this.navPathStack.popToName('navidesnation_add')
        })
        .margin({ top: 60, left: 20 })
      Button('popToIndex')
        .height('5%')
        .onClick(() => { // 点击后滑动到指定位置，即下滑100.0vp的距离，滑动过程配置有动画
          this.navPathStack.popToIndex(0);
        })
        .margin({ top: 110, left: 20 })
      Button('movetoTop')
        .height('5%')
        .onClick(() => { // 点击后回到顶部
            this.navPathStack.moveToTop('navidesnation_add')
        })
        .margin({ top: 150, left: 20 })
      Button('moveIndexToTop')
        .height('5%')
        .onClick(() => { // 点击后滑到下一页
            this.navPathStack.moveIndexToTop(0)
        }).margin({ top: 200, left: 20 })
      Button('getAllPathName')
        .height('5%')
        .onClick(() => { // 点击后滑到下一页
          this.navPathStack.getAllPathName()
        }).margin({ top: 250, left: 20 })
      Button('getParamByIndex')
        .height('5%')
        .onClick(() => { // 点击后滑到下一页
          this.navPathStack.getParamByIndex(0)
        }).margin({ top: 300, left: 20 })
      Button('getParamByName')
        .height('5%')
        .onClick(() => { // 点击后滑到下一页
          this.navPathStack.getParamByName('navidesnation_add')
        }).margin({ top: 350, left: 20 })

      Button('getIndexbyName')
        .height('5%')
        .onClick(() => { // 点击后滑到下一页
          this.navPathStack.getIndexByName('navidesnation_add')
        }).margin({ top: 400, left: 20 })
      Button('pushPaht')
        .height('5%')
        .onClick(() => { // 点击后滑到下一页
          this.navPathStack.pushPath(new NavPathInfo('',''))
        }).margin({ top: 400, left: 20 })


    }.width('100%').height('100%').backgroundColor(0xDCDCDC)


  }
}