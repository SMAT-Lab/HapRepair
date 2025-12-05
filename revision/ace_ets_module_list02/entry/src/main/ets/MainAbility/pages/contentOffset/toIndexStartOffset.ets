/**
 * Copyright (c) 2024 Huawei Device Co., Ltd.
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
//SUB_ACE_UI_COMPONENT_LISTANDGRID_LIST_OFFSET_0140
@Entry
@Component
struct listStartOffset_scrollToIndex {
  scroller: Scroller = new Scroller;
  private arr: number[] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

  build() {
    Column() {
      List({ space: 10, initialIndex: 0, scroller: this.scroller }) {
        ForEach(this.arr, (item: number, index?: number) => {
          ListItem() {
            Text('ListItem' + item)
              .width('90%')
              .height(50)
              .textAlign(TextAlign.Center)
              .borderRadius(10)
              .backgroundColor(0xFFFFFF)
          }.id('List_scrollToIndex8_ListItem_' + item)
        }, (item: string) => item)

      }
      .id('List_scrollToIndex8_List')
      .width('90%')
      .height('40%')
      .backgroundColor(Color.Pink)
      .scrollBar(BarState.On)
      .contentStartOffset(10)
      .contentEndOffset(10)

      Text('scrollToIndex 8')
      Column() {
        Row() {
          Button('AUTO')
            .id('toIndex8_auto')
            .fontSize(14)
            .margin(5)
            .onClick(() => {
              this.scroller.scrollToIndex(8, true, ScrollAlign.AUTO)
            })
        }
      }
      Column() {
        Row() {
          Button('CENTER')
            .id('toIndex8_center')
            .fontSize(14)
            .margin(5)
            .onClick(() => {
              this.scroller.scrollToIndex(8, true, ScrollAlign.CENTER)
            })
        }
      }

    }.width('100%')
    .height('100%')
    .backgroundColor(0xDCDCDC)

  }
}