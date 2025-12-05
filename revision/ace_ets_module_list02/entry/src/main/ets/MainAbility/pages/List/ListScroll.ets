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

@Entry
@Component
struct ListScroll {
  @State listPosition: number = 0;
  private arr: number[] = Array.from(Array(60).keys()).map(item => item)
  private scrollerForList: Scroller = new Scroller()

  build() {
    Column() {
      Row() {
        Button('To(14)')
          .onClick(() => {
            this.scrollerForList.scrollToIndex(14, false)
          })
          .height(45)
          .key('ListScroll_Button_To_14')
        Button('To(15)')
          .onClick(() => {
            this.scrollerForList.scrollToIndex(15, true)
          })
          .height(45)
          .key('ListScroll_Button_To_15')
      }.margin({ top: 10 })

      Row() {
        Button('To(0)')
          .onClick(() => {
            this.scrollerForList.scrollToIndex(0, false)
          })
          .height(45)
          .key('ListScroll_Button_To_0')
        Button('To(max)')
          .onClick(() => {
            this.scrollerForList.scrollToIndex(this.arr.length - 1, true)
          })
          .height(45)
          .key('ListScroll_Button_To_max')
      }.margin({ top: 10 })

      List({ space: 5, initialIndex: 23, scroller: this.scrollerForList }) {
        ForEach(this.arr, (item: number) => {
          ListItem() {
            Text('ListItem' + item)
              .width('100%')
              .height('100%')
              .borderRadius(15)
              .fontSize(16)
              .textAlign(TextAlign.Center)
              .backgroundColor(Color.Gray)
          }
          .width('90%')
          .height(50)
        }, (item: string) => item)
      }
      .listDirection(Axis.Vertical)
      .width('100%')
      .height('50%')
      .margin({ top: 10 })
      .edgeEffect(EdgeEffect.None)
      .alignListItem(ListItemAlign.Center)
      .onScrollIndex((start, end) => {
        let eventData: events_emitter.EventData = {
          data: {
            'start': start,
            'end': end
          }
        }
        if (start === 14) {
          let event: events_emitter.InnerEvent = {
            eventId: 60000,
            priority: events_emitter.EventPriority.LOW
          }
          events_emitter.emit(event, eventData)
        } else if (start === 15) {
          let event: events_emitter.InnerEvent = {
            eventId: 60001,
            priority: events_emitter.EventPriority.LOW
          }
          events_emitter.emit(event, eventData)
        } else if (start === 0) {
          let event: events_emitter.InnerEvent = {
            eventId: 60002,
            priority: events_emitter.EventPriority.LOW
          }
          events_emitter.emit(event, eventData)
        } else if (end === 59) {
          let event: events_emitter.InnerEvent = {
            eventId: 60003,
            priority: events_emitter.EventPriority.LOW
          }
          events_emitter.emit(event, eventData)
        }
      })
      .key('ListScroll_List1')
    }
    .width('100%')
    .height('100%')
  }
}