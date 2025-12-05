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
struct headerStartOffsetExample {
  scroller: ListScroller = new ListScroller()
  @State arr: number[] = [1,2,3]
  @State arr2: number[] = [1,2,3,4]

  @Builder
  itemHead(text: string) {
    if (text === '第2个ListItemGroup'){
      Text(text)
        .id('group_head_startOffset_2')
        .fontSize(20)
        .backgroundColor(Color.Orange)
        .width('100%')
        .padding(10)
    } else {
      Text(text)
        .fontSize(20)
        .backgroundColor(0xAABBCC)
        .width('100%')
        .padding(10)
    }
  }

  build() {
    Column() {
      List({ space: 5, scroller: this.scroller }) {
        ForEach(this.arr,(item0:number, groupIndex: number) => {
          ListItemGroup({ header: this.itemHead('第' + item0 + '个ListItemGroup') }) {
            ForEach(this.arr2, (item: string, itemIndex:number) => {
              ListItem() {
                Text('Group:' + (groupIndex + 1) + ', Item:' + item)
                  .width('100%')
                  .height(100)
                  .fontSize(20)
                  .textAlign(TextAlign.Center)
                  .backgroundColor(Color.Pink)
              }
              .id('group_item_startOffset_' + (groupIndex + 1) + '_' + item)
            }, (item: string) => item)
          }
        })
      }
      .id('group_list_startOffset')
      .width('90%')
      .height('40%')
      .listDirection(Axis.Vertical)
      .scrollBar(BarState.On)
      .sticky(StickyStyle.Header | StickyStyle.Footer)
      .contentStartOffset(10)
      .contentEndOffset(10)
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
      Button('scrollToItemInGroup(1, 2)')
        .margin({ top: 5 })
        .id('startOffset_itemInGroup')
        .onClick(() => {
          this.scroller.scrollToItemInGroup(1, 2)
        })
    }
    .width('100%')
    .height('100%')
    .backgroundColor(0xDCDCDC)
    .margin({ top: 5 })
  }
}