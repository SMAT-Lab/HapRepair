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
struct ListLevel1 {
  private arr1: number[] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  private listScroller = new ListScroller();
  @State listDirection: Axis = Axis.Vertical;
  @State text: number = 0
  @State isShowRect: boolean = false
  @State itemRect: RectResult = { x:0, y:0, width:0, height:0}
  @State isShowOffset: boolean = false
  @State xoffset: number = 0
  @State yoffset: number = 0
  @State num: number = 0
  @State strokeWidth: number = 2.000000;
  @State startMargin: number = 10.000000;
  @State endMargin: number = 10.000000;
  @State contentStartOffset: number = 10.000000;
  @State contentEndOffset: number = 10.000000;
  @State startDeleteAreaDistance: number = 80;
  @State endDeleteAreaDistance: number = 80;


  onPageShow() {
    console.info('[list] page show called');
    let stateChangeEvent: events_emitter.InnerEvent = {
      eventId: 124,
      priority: events_emitter.EventPriority.LOW
    }
    events_emitter.on(stateChangeEvent, this.stateChangCallBack);

    let stateChangeEventTwo: events_emitter.InnerEvent = {
      eventId: 125,
      priority: events_emitter.EventPriority.LOW
    }
    events_emitter.on(stateChangeEventTwo, this.stateChangCallBackTwo);

    let stateChangCallBackThree: events_emitter.InnerEvent = {
      eventId: 126,
      priority: events_emitter.EventPriority.LOW
    }
    events_emitter.on(stateChangCallBackThree, this.stateChangCallBackThree);
  }

  private stateChangCallBack = (eventData: events_emitter.EventData) => {
    console.info('[stateChangCallBack] stateChangCallBack');
    if (eventData != null) {
      console.info('[stateChangCallBack] state change called:' + JSON.stringify(eventData));
      if (eventData.data?.contentStartOffset != null) {
        this.contentStartOffset = eventData.data?.contentStartOffset;
      }
      if (eventData.data?.contentEndOffset != null) {
        this.contentEndOffset = eventData.data?.contentEndOffset;
      }
      if (eventData.data?.listDirection != null) {
        this.listDirection = eventData.data?.listDirection;
      }
      if (eventData.data?.padding != null) {
        this.num = eventData.data?.padding;
      }
    }
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

  private stateChangCallBackThree = (eventData: events_emitter.EventData) => {
    console.info('[stateChangCallBackThree] stateChangCallBack');
    if (eventData != null) {
      console.info('[stateChangCallBackThree] state change called:' + JSON.stringify(eventData));
      if (eventData.data?.startDeleteAreaDistance != null) {
        this.startDeleteAreaDistance = eventData.data?.startDeleteAreaDistance;
      }
      if (eventData.data?.endDeleteAreaDistance != null) {
        this.endDeleteAreaDistance = eventData.data?.endDeleteAreaDistance;
      }
    }
  }

  @Builder pixelMapBuilder() {
    Column() {
      Text(`${this.text}`)
        .width('50%').height(60).fontSize(16).borderRadius(10)
        .textAlign(TextAlign.Center).backgroundColor(Color.Yellow)
    }
  }

  @Builder itemEnd(item:number) {
    Row() {
      Button('删除').margin('4vp').onClick(() => {
        let index = this.arr1.indexOf(item)
        this.arr1.splice(index, 1)
      })
      Button('取消').margin('4vp')
    }.padding('4vp').justifyContent(FlexAlign.SpaceEvenly).width(0)
  }

  build() {
    Scroll(){
      Column() {
        if (this.isShowRect) {
          Text('itemRectY:' + Math.round(this.itemRect.y)).id('itemRectY')
          Text('itemRectWidth:' + Math.round(this.itemRect.width)).id('itemRectWidth')
          Text('itemRectHeight:' + Math.round(this.itemRect.height)).id('itemRectHeight')
        }
        if (this.isShowOffset) {
          Text('xoffset:' + Math.round(this.xoffset)).id('xoffset')
          Text('yoffset:' + Math.round(this.yoffset)).id('yoffset')
        }

        Button('GetItemRect')
          .onClick(() => {
            this.itemRect = this.listScroller.getItemRect(0)
            this.isShowRect = true
          })
          .id('GetItemRectButton')
        Button('currentOffset')
          .onClick(() => {
            this.xoffset = this.listScroller.currentOffset().xOffset;
            this.yoffset = this.listScroller.currentOffset().yOffset;
            this.isShowOffset = true
          })
          .id('currentOffsetButton')
        
        List({ space: 10, initialIndex: 0, scroller: this.listScroller }) {
          ForEach(this.arr1, (item: number) => {
            ListItem() {
              Text('item' + item)
                .width(200)
                .height(100)
                .fontSize(16)
                .textAlign(TextAlign.Center)
                .borderRadius(10)
                .backgroundColor(0xFFFFFF)
            }
            .swipeAction({
              start: {
                builder: () => {
                  this.itemEnd(item)
                },
                actionAreaDistance: this.startDeleteAreaDistance,
                onAction: () => {
                  animateTo({ duration: 1000 }, () => {
                    let index = this.arr1.indexOf(item)
                    this.arr1.splice(index, 1)
                  })
                }
              },
              end: {
                builder: () => {
                  this.itemEnd(item)
                },
                actionAreaDistance: this.endDeleteAreaDistance,
                onAction: () => {
                  animateTo({ duration: 1000 }, () => {
                    let index = this.arr1.indexOf(item)
                    this.arr1.splice(index, 1)
                  })
                }
              },
              edgeEffect: SwipeEdgeEffect.None
            })
            .id('listitemlevel' + item)
          }, (item: string) => item)
        }
        .id('listlevel1')
        .padding(this.num)
        .listDirection(this.listDirection)
        .width('90%')
        .height('80%')
        .divider({
          strokeWidth: this.strokeWidth,
          color: 0xFFFFFF,
          startMargin: this.startMargin,
          endMargin: this.endMargin
        })
        .edgeEffect(EdgeEffect.None)
        .chainAnimation(false)
        .contentStartOffset(this.contentStartOffset)
        .contentEndOffset(this.contentEndOffset)
      }.width('100%').height('100%')
    }
  }
}