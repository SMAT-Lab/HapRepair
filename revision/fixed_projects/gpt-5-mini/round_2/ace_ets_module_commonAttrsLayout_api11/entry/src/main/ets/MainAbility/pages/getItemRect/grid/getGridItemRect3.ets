
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
struct getGridItemRect3 {
  scroller: Scroller = new Scroller()
  // columnsGap1 未发生变化，改为普通变量
  columnsGap1: number = 0
  // rowsGap1 未发生变化，改为普通变量
  rowsGap1: number = 0
  @State x1: number = 0
  @State y1: number = 0
  @State index: number = 0
  @State x: number = 0
  @State y: number = 0
  @State width1: number = 0
  @State height1: number = 0
  messageManager: MessageManager = new MessageManager()

  onPageShow() {
    console.info('getGridItemRect3 onPageShow');
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
    Column({ space: 5 }) {
      Button('点击获取信息').onClick((event: ClickEvent) => {
        this.x = this.scroller.getItemRect(this.index).x
        this.y = this.scroller.getItemRect(this.index).y
        this.width1 = this.scroller.getItemRect(this.index).width
        this.height1 = this.scroller.getItemRect(this.index).height
      }).id('getGridItemRect3_button')
      Text('' + this.x).key('getGridItemRect3_text1')
      Text('' + this.y).key('getGridItemRect3_text2')
      Text('' + this.width1).key('getGridItemRect3_text3')
      Text('' + this.height1).key('getGridItemRect3_text4')

      Grid(this.scroller) {
        GridItem() {
          Text('0')
            .fontSize(16)
            .backgroundColor(0xF9CF93)
            .width('100%')
            .height('100%')
            .textAlign(TextAlign.Center)
        }
        .rowStart(0)
        .rowEnd(1)
        .border({ width: 1 })
        .offset({ x: this.x1, y: this.y1 })
        .key('getGridItemRect3_gridItem_0')

        GridItem() {
          Text('1')
            .fontSize(16)
            .backgroundColor(0xF9CF93)
            .width('100%')
            .height('100%')
            .textAlign(TextAlign.Center)
        }
        .columnStart(0)
        .columnEnd(1)
        .border({ width: 1 })
        .key('getGridItemRect3_gridItem_1')


        GridItem() {
          Text('2')
            .fontSize(16)
            .backgroundColor(0xF9CF93)
            .width('100%')
            .height('100%')
            .textAlign(TextAlign.Center)
        }
        .border({ width: 1 })
        .key('getGridItemRect3_gridItem_2')

        GridItem() {
          Text('3')
            .fontSize(16)
            .backgroundColor(0xF9CF93)
            .width('100%')
            .height('100%')
            .textAlign(TextAlign.Center)
        }
        .border({ width: 1 })
        .key('getGridItemRect3_gridItem_3')

        GridItem() {
          Text('4')
            .fontSize(16)
            .backgroundColor(0xF9CF93)
            .width('100%')
            .height('100%')
            .textAlign(TextAlign.Center)
        }
        .border({ width: 1 })

        GridItem() {
          Text('5')
            .fontSize(16)
            .backgroundColor(0xF9CF93)
            .width('100%')
            .height('100%')
            .textAlign(TextAlign.Center)
        }
        .border({ width: 1 })

        GridItem() {
          Text('6')
            .fontSize(16)
            .backgroundColor(0xF9CF93)
            .width('100%')
            .height('100%')
            .textAlign(TextAlign.Center)
        }
        .border({ width: 1 })

        GridItem() {
          Text('7')
            .fontSize(16)
            .backgroundColor(0xF9CF93)
            .width('100%')
            .height('100%')
            .textAlign(TextAlign.Center)
        }
        .border({ width: 1 })

        GridItem() {
          Text('8')
            .fontSize(16)
            .backgroundColor(0xF9CF93)
            .width('100%')
            .height('100%')
            .textAlign(TextAlign.Center)
        }
        .border({ width: 1 })

        GridItem() {
          Text('9')
            .fontSize(16)
            .backgroundColor(0xF9CF93)
            .width('100%')
            .height('100%')
            .textAlign(TextAlign.Center)
        }
        .border({ width: 1 })

      }
      .columnsTemplate('1fr 1fr 1fr 1fr 1fr')
      .rowsTemplate('1fr 1fr 1fr 1fr 1fr')
      .columnsGap(this.columnsGap1)
      .rowsGap(this.rowsGap1)
      .width(300)
      .backgroundColor(0xFAEEE0)
      .height(500)

    }.width('100%').margin({ top: 5 })
  }
}