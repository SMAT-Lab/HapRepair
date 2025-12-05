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
struct getScrollItemRect3 {
  scroller: Scroller = new Scroller()
  @State arr: number[] = [0, 1, 2, 3, 4, 5]
  @State x1: number = 0
  @State y1: number = 0
  @State index: number = 0
  @State x: number = 0
  @State y: number = 0
  @State width1: number = 0
  @State height1: number = 0
  messageManager: MessageManager = new MessageManager()

  onPageShow() {
    console.info('getScrollItemRect3 onPageShow');
    globalThis.value = {
      name: 'messageManager', message: this.messageManager
    }
    let callback: Callback = (message: ESObject) => {
      console.error('message = ' + message.name + "--" + message.value);
      if (message.name == 'index') {
        this.index = message.value;
      } else if (message.name == 'x1') {
        this.x1 = message.value;
      } else if (message.name == 'y1') {
        this.y1 = message.value;
      }
    }
    this.messageManager.registerCallback(callback);
  }

  build() {
    Column() {
      Button('点击获取信息').onClick((event: ClickEvent) => {
        this.x = this.scroller.getItemRect(this.index).x
        this.y = this.scroller.getItemRect(this.index).y
        this.width1 = this.scroller.getItemRect(this.index).width
        this.height1 = this.scroller.getItemRect(this.index).height
      }).id('getScrollItemRect3_button')
      Text('' + this.x).key('getScrollItemRect3_text1')
      Text('' + this.y).key('getScrollItemRect3_text2')
      Text('' + this.width1).key('getScrollItemRect3_text3')
      Text('' + this.height1).key('getScrollItemRect3_text4')

      Scroll(this.scroller) {
        Column() {
          ForEach(this.arr, (item: number) => {
            Text(item.toString())
              .width('100%')
              .height(100)
              .backgroundColor('#3366CC')
              .borderRadius(15)
              .fontSize(16)
              .textAlign(TextAlign.Center)
          })
        }
        .width(300)
        .position({ x: this.x1, y: this.y1 })

      }
      .scrollable(ScrollDirection.Vertical)
      .scrollBar(BarState.Auto)
      .height(500)

    }
  }
}