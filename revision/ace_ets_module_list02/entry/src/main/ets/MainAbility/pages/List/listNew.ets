/*
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
import events_emitter from '@ohos.events.emitter';
import Log from '../../common/Log';

const TAG = 'ets_apiLack_add';

@Entry
@Component
struct ListNewLanes {
  private arr: number[] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  @State listPosition: number = 0 // 0代表滚动到List顶部，1代表中间值，2代表滚动到List底
  @State edgeEffect: EdgeEffect = EdgeEffect.None;
  @State alwaysEnabled: boolean = true
  private scroller: Scroller = new Scroller()

  aboutToAppear() {
    Log.showInfo(TAG, `aboutToAppear ListLanes start`)
  }

  aboutToDisappear() {
    Log.showInfo(TAG, `aboutToDisAppear ListLanes end`)
  }

  private stateChangCallBack = (eventData: events_emitter.EventData) => {
    console.info('list page state change called:' + JSON.stringify(eventData));
    if (eventData != null) {
      if (eventData.data?.alwaysEnabled != null) {
        this.alwaysEnabled = eventData.data?.alwaysEnabled;
      }
    }
  }

  onPageShow() {
    console.info('list page show called');
    let stateChangeEvent: events_emitter.InnerEvent = {
      eventId: 322,
      priority: events_emitter.EventPriority.LOW
    }
    events_emitter.on(stateChangeEvent, this.stateChangCallBack);
  }

  build() {
    Flex({ direction: FlexDirection.Column, alignItems: ItemAlign.Center, justifyContent: FlexAlign.Center }) {

      Text('list-Lanes')
        .width(100)
        .height(70)
        .fontSize(20)
        .opacity(1)
        .align(Alignment.TopStart)
        .fontColor(0xCCCCCC)
        .lineHeight(25)
        .border({ width: 1 })
        .padding(10)
        .textAlign(TextAlign.Center)
        .textOverflow({ overflow: TextOverflow.None })
        .key('lanesText')

      Text('list-AlignListItem')
        .width(100)
        .height(70)
        .fontSize(20)
        .opacity(1)
        .align(Alignment.TopStart)
        .fontColor(0xCCCCCC)
        .lineHeight(25)
        .border({ width: 1 })
        .padding(10)
        .textAlign(TextAlign.Center)
        .textOverflow({ overflow: TextOverflow.None })
        .key('alignListItemText')

      Text('list-OnScrollBegin')
        .width(100)
        .height(70)
        .fontSize(20)
        .opacity(1)
        .align(Alignment.TopStart)
        .fontColor(0xCCCCCC)
        .lineHeight(25)
        .border({ width: 1 })
        .padding(10)
        .textAlign(TextAlign.Center)
        .textOverflow({ overflow: TextOverflow.None })
        .key('onScrollBeginText')

      Stack({ alignContent: Alignment.TopStart }) {
        Scroll(this.scroller) {
          List({ space: 20, initialIndex: 0 }) {
            ForEach(this.arr, (item: number) => {
              ListItem() {
                Text('' + item)
                  .width('100%')
                  .height(100)
                  .fontSize(18)

              }
              .border({ width: 2, color: Color.Green })
            }, (item: string) => item)
          }
          .key('list_new1')
          .height(300)
          .width('90%')
          .editMode(true)
          .alignListItem(ListItemAlign.Start)
          .border({ width: 3, color: Color.Red })
          .edgeEffect(this.edgeEffect, { alwaysEnabled: this.alwaysEnabled })
          .lanes(4)
          .onScrollFrameBegin((offset: number, state: ScrollState) => {
            if ((this.listPosition == 0 && offset >= 0) || (this.listPosition == 2 && offset <= 0)) {
              this.scroller.scrollBy(0, -offset)
              return { offsetRemain: 0 }
            }
            this.listPosition = 1;
            return { offsetRemain: offset }

            try {
              let backData: events_emitter.EventData = {
                data: {
                  'result': 'success'
                }
              }
              let backEvent: events_emitter.InnerEvent = {
                eventId: 60216,
                priority: events_emitter.EventPriority.LOW
              }
              console.info('onScrollBegin onClick start to emit action state')
              events_emitter.emit(backEvent, backData)
            } catch (err) {
              console.info('onScrollBegin onClick emit action state err: ' + JSON.stringify((err as Error)?.message))
            }
          })
        }
        .scrollable(ScrollDirection.Vertical)
        .scrollBar(BarState.On)
        .scrollBarColor(Color.Gray)
        .scrollBarWidth(30)
        .onScroll((xOffset: number, yOffset: number) => {
          console.info(xOffset + ' ' + yOffset)
        })
        .onScrollEdge((side: Edge) => {
          console.info('To the edge')
        })
        .onScrollEnd(() => {
          console.info('Scroll Stop')
        })

        Button('scroll 100')

          .onClick(() => { // 点击后下滑100.0距离
            this.scroller.scrollTo({ xOffset: 0, yOffset: this.scroller.currentOffset().yOffset + 100 })
          })
          .margin({ top: 10, left: 20 })
        Button('back top')
          .onClick(() => { // 点击后回到顶部
            this.scroller.scrollEdge(Edge.Top)
          })
          .margin({ top: 60, left: 20 })
        Button('next page')
          .key('onScrollBegin')
          .onClick(() => { // 点击后下滑到底部
            this.scroller.scrollPage({ next: true })
          })
          .margin({ top: 110, left: 20 })
      }.width('100%').height('100%').backgroundColor(0xDCDCDC)

    }.width('100%').height('100%')
  }
}

