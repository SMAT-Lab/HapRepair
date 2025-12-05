
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
@Entry
@Component
struct paddingStartOffsetExample {
  scroller: Scroller = new Scroller()
  @State arr: number[] = [1, 2, 3, 4, 5, 6, 7, 8, 9]

  build() {
    Column() {
      List({ space: 10, scroller: this.scroller }) {
        ForEach(this.arr,(item:number, index: number) => {
          ListItem() {
            Text('item' + item)
              .width('90%')
              .height(100)
              .fontSize(16)
              .borderRadius(10)
              .textAlign(TextAlign.Center)
              .backgroundColor(0xFFFFFF)
          }
          .id('listItem_padStart_' + item)
        }, (item: string) => item)

      }
      .id('list_padStart')
      .width('100%')
      .height('40%')
      .padding(5)
      .contentStartOffset(5)
      .listDirection(Axis.Vertical)
      .scrollBar(BarState.Off)
      .backgroundColor(0xDCDCDC)
      .divider({ strokeWidth: 2, color: 0xFFFFFF, startMargin: 20, endMargin: 20 }) // 每行之间的分界线
      .edgeEffect(EdgeEffect.Spring) // 边缘效果设置为Spring
      .onScrollIndex((firstIndex: number, lastIndex: number, centerIndex: number) => {
        console.info('first' + firstIndex)
        console.info('last' + lastIndex)
        console.info('center' + centerIndex)
      })
      .onScrollVisibleContentChange((start: VisibleListContentInfo, end: VisibleListContentInfo) => {
        console.log(' start index: ' + start.index +
          ' start item group area: ' + start.itemGroupArea +
          ' start index in group: ' + start.itemIndexInGroup)
        console.log(' end index: ' + end.index +
          ' end item group area: ' + end.itemGroupArea +
          ' end index in group: ' + end.itemIndexInGroup)
      })
    }
    .width('100%')
    .height('100%')
    .backgroundColor(0xDCDCDC)
    .margin({ top: 5 })
  }
}