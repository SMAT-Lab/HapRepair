/**
 * Copyright (c) 2022 Shenzhen Kaihong Digital Industry Development Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the 'License');
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an 'AS IS' BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import Log from '../../common/Log';
import events_emitter from '@ohos.events.emitter';

const TAG = 'ets_apiLack_add';

@Entry
@Component
struct ListIdle {
  private arr: number[] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  @State editFlag: boolean = false
  private scroller: Scroller = new Scroller()
  @State onScroll: boolean = false

  aboutToAppear() {
    Log.showInfo(TAG, `aboutToAppear ListIdle start`)
  }

  aboutToDisappear() {
    Log.showInfo(TAG, `aboutToDisAppear ListIdle end`)
  }

  build() {
    Flex({ direction: FlexDirection.Column, alignItems: ItemAlign.Center, justifyContent: FlexAlign.Center }) {

      Text('list-Idle')
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
        .key('IdleText')

      Text('list-Scroll')
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
        .key('ScrollText')

      Text('list-Fling')
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
        .key('FlingText')

      Stack({ alignContent: Alignment.TopStart }) {
        Scroll(this.scroller) {
          List({ space: 20, initialIndex: 0 }) {
            ForEach(this.arr, (item: string) => {
              ListItem() {
                Text('' + item)
                  .width('100%')
                  .height(100)
                  .fontSize(16)
                  .textAlign(TextAlign.Center)
                  .borderRadius(10)
                  .backgroundColor(0xFFFFFF)
              }.editable(true)
            }, (item: string) => item)

          }
          .editMode(true)
          .key('ScrollList')
          .listDirection(Axis.Vertical) // 排列方向
          .divider({ strokeWidth: 2, color: 0xFFFFFF, startMargin: 20, endMargin: 20 }) // 每行之间的分界线
          .edgeEffect(EdgeEffect.None) // 滑动到边缘无效果
          .chainAnimation(false) // 联动特效关闭
          .onScroll((scrollOffset: number, scrollState: ScrollState) => {
            console.info('scrollOffset' + scrollOffset)
            console.info('scrollState   ' + ScrollState.Scroll)
            console.info('scrollState   ' + ScrollState.Idle)
            console.info('scrollState   ' + ScrollState.Fling)
          })
          .onClick(() => {
            try {
              let backData: events_emitter.EventData = {
                data: {
                  'State': this.onScroll
                }
              }
              let backEvent: events_emitter.InnerEvent = {
                eventId: 60303,
                priority: events_emitter.EventPriority.LOW
              }
              console.info('Scroll start to emit action state')
              events_emitter.emit(backEvent, backData)
            } catch (err) {
              console.info('Scroll emit action state err: ' + JSON.stringify((err as Error)?.message))
            }
            try {
              this.scroller.scrollPage({ next: true })
              this.onScroll = true
            } catch (err) {
              console.info('Scroll emit action state err: ' + JSON.stringify((err as Error)?.message))
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
      }
    }
    .width('100%')
    .height('100%')
  }
}

