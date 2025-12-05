/*
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

import events_emitter from '@ohos.events.emitter';

@Entry
@Component
struct ListLevel1Horizontal {
  private arr1: number[] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  @State strokeWidth: number = 2.000000;
  @State startMargin: number = 10.000000;
  @State endMargin: number = 10.000000;

  onPageShow() {
    console.info('[list] page show called');
    let stateChangeEventTwo: events_emitter.InnerEvent = {
      eventId: 125,
      priority: events_emitter.EventPriority.LOW
    }
    events_emitter.on(stateChangeEventTwo, this.stateChangCallBackTwo);
  }

  private stateChangCallBackTwo = (eventData: events_emitter.EventData) => {
    console.info('[stateChangCallBackTwo] stateChangCallBack');
    if (eventData != null) {
      console.info('[stateChangCallBackTwo] state change called:' + JSON.stringify(eventData));
      if (eventData.data?.strokeWidth != null) {
        this.strokeWidth = eventData.data?.strokeWidth;
      }
      if (eventData.data?.startMargin != null) {
        this.startMargin = eventData.data?.startMargin;
      }
      if (eventData.data?.endMargin != null) {
        this.endMargin = eventData.data?.endMargin;
      }
    }
  }

  build() {
    Stack() {
      Column() {
        List({ space: 20, initialIndex: 0 }) {
          ForEach(this.arr1, (item:number) => {
            ListItem() {
              Text('' + item)
                .width('100%')
                .height(100)
                .fontSize(16)
                .textAlign(TextAlign.Center)
                .borderRadius(10)
                .backgroundColor(0xFFFFFF)
            }
          }, (item:string) => item)
        }
        .id('listlevel1Horizontal')
        .listDirection(Axis.Horizontal)
        .height(300)
        .divider({
          strokeWidth: this.strokeWidth,
          color: 0xFFFFFF,
          startMargin: this.startMargin,
          endMargin: this.endMargin
        })
      }.width('100%')
    }.width('100%').height('50%').backgroundColor(0xDCDCDC).padding({ top: 5 })
  }
}