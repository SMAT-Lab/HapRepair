
/**
 * Copyright (c) 2023 Huawei Device Co., Ltd.
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
import {MessageManager,Callback} from '../../../common/MessageManager';

@Entry
@Component
struct getListItemRect1 {
  private arr: number[] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
  private listScroller: ListScroller = new ListScroller()
  // space1 未发生变化，改为普通变量
  space1: number = 20
  @State padding1: number = 0
  @State margin1: number = 0
  @State index: number = 0
  @State x: number = 0
  @State y: number = 0
  @State width1: number = 0
  @State height1: number = 0
  messageManager: MessageManager = new MessageManager()

  onPageShow() {
    console.info('getListItemRect1 onPageShow');
    globalThis.value = {
      name: 'messageManager', message: this.messageManager
    }
    let callback: Callback = (message: ESObject) => {
      console.error('message = ' + message.name + "--" + message.value);
      if (message.name == 'index') {
        this.index = message.value;
      } else if (message.name == 'padding1') {
        this.padding1 = message.value;
      } else if (message.name == 'margin1') {
        this.margin1 = message.value;
      }
    }
    this.messageManager.registerCallback(callback);
  }

  build() {
    Column() {
      Button('点击获取信息').onClick((event: ClickEvent) => {
        this.x = this.listScroller.getItemRect(this.index).x
        this.y = this.listScroller.getItemRect(this.index).y
        this.width1 = this.listScroller.getItemRect(this.index).width
        this.height1 = this.listScroller.getItemRect(this.index).height
      }).id('getListItemRect1_button')
      Text('' + this.x).key('getListItemRect1_text1')
      Text('' + this.y).key('getListItemRect1_text2')
      Text('' + this.width1).key('getListItemRect1_text3')
      Text('' + this.height1).key('getListItemRect1_text4')

      Button('点击滚动到底部').onClick((event: ClickEvent) => {
        this.listScroller.scrollEdge(Edge.End)
      }).id('getListItemRect1_button2')
      Button('还原').onClick((event: ClickEvent) => {
        this.index = 0
        this.padding1 = 0
        this.margin1 = 0
        this.listScroller.scrollEdge(Edge.Start)
      }).id('getListItemRect1_button3')

      List({ space: this.space1, initialIndex: 0, scroller: this.listScroller }) {
        ForEach(this.arr, (item: number) => {
          ListItem() {
            Text('' + item)
              .width('100%')
              .height('100')
              .fontSize(16)
              .textAlign(TextAlign.Center)
              .borderRadius(10)
              .backgroundColor(0xFFFFFF)
              .key('getListItemRect1_listItem_' + item)
          }
          .width('100%').height(100)
          .margin(this.margin1)
        }, (item: string) => item)
      }
      .key('getListItemRect1_list1')
      .listDirection(Axis.Vertical) // 排列方向
      .scrollBar(BarState.Off)
      .edgeEffect(EdgeEffect.Spring) // 边缘效果设置为Spring
      .width(300)
      .height('60%')
      .padding(this.padding1)

    }
    .width('100%')
    .height('100%')
    .backgroundColor(0xDCDCDC)
    .margin(5)

  }
}