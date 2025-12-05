/**
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
struct listDidExample {
  scroller: Scroller = new Scroller()
  private arr: number[] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  @State tempOffset: number = 1;

  build() {
    Column() {
      Text('scrollState: ' + this.tempOffset)
        .id('list_onDidScroll')
      List({ space: 20, initialIndex: 0 }) {
        ForEach(this.arr, (item: number) => {
          ListItem() {
            Text('' + item)
              .width('100%').height(100).fontSize(16)
              .textAlign(TextAlign.Center).borderRadius(10).backgroundColor(0xFFFFFF)
          }
        }, (item: string) => item)
      }
      .id('scrollSource_list')
      .onDidScroll((scrollOffset: number, scrollState: ScrollState) => {
        console.info(`onDidScroll scrollState = ${scrollState[scrollState]}, offset = ${scrollOffset}`)
        this.tempOffset = scrollState
        if (scrollState == ScrollState.Idle) {
          console.info('ScrollState.Idle: ' + this.tempOffset)
        }
      })
      .listDirection(Axis.Vertical) // 排列方向
      .scrollBar(BarState.Off)
      .friction(0.6)
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
      .width('90%')
      .height('60%')
      .onWillScroll((scrollOffset: number, scrollState: ScrollState, scrollSource: ScrollSource) => {
        console.info('scrollSource: ' + scrollSource)
        console.info(`onWillScroll scrollState = ${scrollState[scrollState]}, offset = ${scrollOffset}`)
      })
      .onScrollStop(() => {
        console.info('Scroll Stop')
      })

    }
    .width('100%')
    .height('100%')
    .backgroundColor(0xDCDCDC)
  }
}