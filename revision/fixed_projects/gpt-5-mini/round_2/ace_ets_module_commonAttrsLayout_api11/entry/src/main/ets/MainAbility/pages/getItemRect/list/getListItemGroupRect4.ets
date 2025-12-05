
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
struct getListItemGroupRect4 {
  private arr: number[] = [1, 2, 3, 4, 5, 6, 7]
  private listScroller: ListScroller = new ListScroller()
  // space1 未发生变化，改为普通变量
  space1: number = 20
  // space2 未发生变化，改为普通变量
  space2: number = 0
  @State index: number = 0
  @State indexInGroup: number = 0
  @State visibility1: Visibility = Visibility.Visible
  @State x: number = 0
  @State y: number = 0
  @State width1: number = 0
  @State height1: number = 0

  @Builder
  itemHead(text: string) {
    Text(text)
      .fontSize(20)
      .backgroundColor(0xAABBCC)
      .width('100%')
      .height(50)
  }

  @Builder
  itemFoot(num: number) {
    Text('共' + num + '个')
      .fontSize(20)
      .backgroundColor(0xAABBCC)
      .width('100%')
      .height(50)
  }

  messageManager: MessageManager = new MessageManager()

  onPageShow() {
    console.info('getListItemGroupRect4 onPageShow');
    globalThis.value = {
      name: 'messageManager', message: this.messageManager
    }
    let callback: Callback = (message: ESObject) => {
      console.error('message = ' + message.name + "--" + message.value);
      if (message.name == 'index') {
        this.index = message.value;
      } else if (message.name == 'indexInGroup') {
        this.indexInGroup = message.value;
      } else if (message.name == 'visibility1') {
        this.visibility1 = message.value;
      }
    }
    this.messageManager.registerCallback(callback);
  }

  build() {
    Column() {
      Button('点击获取信息').onClick((event: ClickEvent) => {
        this.x = this.listScroller.getItemRectInGroup(this.index, this.indexInGroup).x
        this.y = this.listScroller.getItemRectInGroup(this.index, this.indexInGroup).y
        this.width1 = this.listScroller.getItemRectInGroup(this.index, this.indexInGroup).width
        this.height1 = this.listScroller.getItemRectInGroup(this.index, this.indexInGroup).height
      }).id('getListItemGroupRect4_button')
      Text('' + this.x).key('getListItemGroupRect4_text1')
      Text('' + this.y).key('getListItemGroupRect4_text2')
      Text('' + this.width1).key('getListItemGroupRect4_text3')
      Text('' + this.height1).key('getListItemGroupRect4_text4')
      Button('Hidden').onClick(() => {
        this.visibility1 = Visibility.Hidden
      }).id('getListItemGroupRect4_button2')
      Button('None').onClick(() => {
        this.visibility1 = Visibility.None
      }).id('getListItemGroupRect4_button3')

      List({ space: this.space1, initialIndex: 0, scroller: this.listScroller }) {
        ListItemGroup({ header: this.itemHead('0'), footer: this.itemFoot(this.arr.length), space: this.space2 }) {
          ListItem() {
            Text('listItemGroup0')
              .width('100%')
              .height("100%")
              .fontSize(16)
              .borderRadius(10)
              .textAlign(TextAlign.Center)
              .backgroundColor('#3366CC')
          }
          .width('100%')
          .height(100)
          .visibility(this.visibility1)
          .key('getListItemGroupRect4_listItemGroup_0')

          ForEach(this.arr, (item: number) => {
            ListItem() {
              Text('listItemGroup' + item)
                .width('100%')
                .height("100%")
                .fontSize(16)
                .borderRadius(10)
                .textAlign(TextAlign.Center)
                .backgroundColor('#3366CC')
            }
            .width('100%')
            .height(100)
            .key('getListItemGroupRect4_listItemGroup_' + item)
          })
        }
        .key('getListItemGroupRect4_listItemGroup_1')
      }
      .key('getListItemGroupRect4_list_1')
      .listDirection(Axis.Vertical) // 排列方向
      .scrollBar(BarState.Off)
      .edgeEffect(EdgeEffect.Spring) // 边缘效果设置为Spring
      .width(300)
      .height('60%')

    }
    .width('100%')
    .height('100%')
    .backgroundColor(0xDCDCDC)
    .margin(5)
  }
}