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
struct singleListHorizontalExample {
  scroller: ListScroller = new ListScroller()
  @State arr: number[] = [1,2]
  @State arr2: number[] = [1,2,3,4,5]

  build() {
    Column() {
      Button('scrollToItemInGroup(1, 2)')
        .id('scroller_group_hoz')
        .onClick(() => {
          this.scroller.scrollToItemInGroup(1, 2)
        })
      List({ space: 5, scroller: this.scroller }) {
        ForEach(this.arr,(item0:number, groupIndex: number) => {
          ListItemGroup() {
            ForEach(this.arr2, (item: string, itemIndex:number) => {
              ListItem() {
                Text('Group:' + (groupIndex + 1) + ', Item:' + item)
                  .width('100%')
                  .height(100)
                  .fontSize(20)
                  .textAlign(TextAlign.Center)
                  .backgroundColor(Color.Pink)
              }
              .id('group_hoz_item')
            }, (item: string) => item)
          }
        })
      }
      .id('group_hoz_list')
      .width('90%')
      .height('40%')
      .listDirection(Axis.Horizontal) // 横向排列方向
      .scrollBar(BarState.Off)
      .friction(0.6)
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
      .onDidScroll((scrollOffset: number, scrollState: ScrollState) => {
        console.info(`onScroll scrollState = ScrollState` + scrollState + `, scrollOffset = ` + scrollOffset)
      })
    }
    .width('100%')
    .borderWidth(2)
    .borderColor(Color.Blue)
    .backgroundColor(0xDCDCDC)
    .margin({ top: 5 })
  }
}