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
import MyUtils from '../../common/MyUtils';
import events_emitter from '@ohos.events.emitter';

@Entry
@Component
struct ListSwipe {
  private listScroller = new ListScroller();
  @State list1Item0State: string = ''
  @State list1Item1Offset: string = ''
  @State closeButtonContent: string = ''
  @State list2Item0State: string = ''
  @State stateContent: string = ''
  @State offsetContent: string = ''

  @Builder
  itemStart() {
    Row() {
      Button('Del').margin('4vp')
      Button('Set').margin('4vp')
    }.padding('4vp').justifyContent(FlexAlign.SpaceEvenly)
  }

  @Builder
  itemEnd() {
    Row() {
      Button('Del').margin('4vp')
      Button('Set').margin('4vp')
    }.padding('4vp').justifyContent(FlexAlign.SpaceEvenly)
  }

  build() {
    Column({ space: 20 }) {
      List({ space: 20, initialIndex: 0 }) {
        ListItem() {
          Text('ListItem0')
            .width('100%')
            .height(30)
            .fontSize(16)
            .textAlign(TextAlign.Center)
            .borderRadius(10)
            .backgroundColor(0xFFFFFF)
        }
        .key('MyList1_ListItem0')
        .transition({ type: TransitionType.Delete, opacity: 0 })
        .swipeAction({
          end: {
            builder: (): void => this.itemEnd(),
            // without onAction, no deleteArea
            onStateChange: (state: SwipeActionState) => {
              if (state == SwipeActionState.COLLAPSED) {
                console.info('MyList1_ListItem0 swipe action state COLLAPSED')
                this.list1Item0State = 'end COLLAPSED'
              } else if (state == SwipeActionState.EXPANDED) {
                console.info('MyList1_ListItem0 swipe action state EXPANDED')
                this.list1Item0State = 'end EXPANDED'
              } else if (state == SwipeActionState.ACTIONING) {
                console.info('MyList1_ListItem0 swipe action state ACTIONING')
                this.list1Item0State = 'end ACTIONING'
              } else {
                console.info('MyList1_ListItem0 swipe action state UNKOWN')
                this.list1Item0State = 'end UNKOWN'
              }
            }
          },
          edgeEffect: SwipeEdgeEffect.Spring,
          onOffsetChange: (offset: number) => {
            if (offset > 0) {
              console.info('MyList1_Item0 swipe start')
              this.offsetContent = 'swipe start'
            } else {
              console.info('MyList1_Item0 swipe end')
              this.offsetContent = 'swipe end'
            }
          }
        })

        ListItem() {
          Text('ListItem1')
            .width('100%')
            .height(30)
            .fontSize(16)
            .textAlign(TextAlign.Center)
            .borderRadius(10)
            .backgroundColor(0xFFFFFF)
        }
        .key('MyList1_ListItem1')
        .transition({ type: TransitionType.Delete, opacity: 0 })
        .swipeAction({
          end: {
            builder: (): void => this.itemEnd(),
            // without onAction, no deleteArea
          },
          edgeEffect: SwipeEdgeEffect.Spring,
          onOffsetChange: (offset: number) => {
            if (offset > 0) {
              console.info('MyList1_Item0 swipe start')
              this.list1Item1Offset = 'swipe start'
            } else {
              console.info('MyList1_Item0 swipe end')
              this.list1Item1Offset = 'swipe end'
            }
          }
        })
      }
      .margin({ top: 5 })
      .listDirection(Axis.Vertical)
      .key('MyList1_swipe')
      .width('90%')

      List({ space: 20, initialIndex: 0, scroller: this.listScroller }) {
        ListItem() {
          Text('ListItem0')
            .width('100%')
            .height(30)
            .fontSize(16)
            .textAlign(TextAlign.Center)
            .borderRadius(10)
            .backgroundColor(0xFFFFFF)
        }
        .key('MyList2_ListItem0')
        .transition({ type: TransitionType.Delete, opacity: 0 })
        .swipeAction({
          start: {
            builder: (): void => this.itemStart(),
            actionAreaDistance: 5,
            onStateChange: (state: SwipeActionState) => {
              if (state == SwipeActionState.COLLAPSED) {
                console.info('MyList2_ListItem0 swipe action state COLLAPSED')
                this.list2Item0State = 'COLLAPSED'
              } else if (state == SwipeActionState.EXPANDED) {
                console.info('MyList2_ListItem0 swipe action state EXPANDED')
                this.list2Item0State = 'EXPANDED'
              } else if (state == SwipeActionState.ACTIONING) {
                console.info('MyList2_ListItem0 swipe action state ACTIONING')
                this.list2Item0State = 'ACTIONING'
              } else {
                console.info('MyList2_ListItem0 swipe action state UNKOWN')
                this.list2Item0State = 'UNKOWN'
              }
            }
          },
          onOffsetChange: (offset: number) => {
            if (offset > 0) {
              MyUtils.sendEvent('swipe start', 70000)
            } else {
              MyUtils.sendEvent('swipe end', 70000)
            }
          },
          edgeEffect: SwipeEdgeEffect.Spring
        })

        ListItem() {
          Text('ListItem1')
            .width('100%')
            .height(30)
            .fontSize(16)
            .textAlign(TextAlign.Center)
            .borderRadius(10)
            .backgroundColor(0xFFFFFF)
        }
        .key('MyList2_ListItem1')
        .transition({ type: TransitionType.Delete, opacity: 0 })
        .swipeAction({
          start: {
            builder: (): void => this.itemEnd(),
            actionAreaDistance: 5,
            // with onAction
            onAction: () => {
            },
            onStateChange: (state: SwipeActionState) => {
              if (state == SwipeActionState.COLLAPSED) {
                console.info('MyList2_ListItem1 swipe action state COLLAPSED')
                this.stateContent = 'COLLAPSED'
              } else if (state == SwipeActionState.EXPANDED) {
                console.info('MyList2_ListItem1 swipe action state EXPANDED')
                this.stateContent = 'EXPANDED'
              } else if (state == SwipeActionState.ACTIONING) {
                console.info('MyList2_ListItem1 swipe action state ACTIONING')
                this.stateContent = 'ACTIONING'
              } else {
                console.info('MyList2_ListItem1 swipe action state UNKOWN')
                this.stateContent = 'UNKOWN'
              }
            }
          },
          onOffsetChange: (offset: number) => {
            if (offset > 0) {
              this.offsetContent = 'swipe start'
            } else {
              this.offsetContent = 'swipe left'
            }
          },
          edgeEffect: SwipeEdgeEffect.Spring
        })
      }
      .divider({ strokeWidth: 2, color: 0xFF, startMargin: 20, endMargin: 20 })
      .margin({ top: 5 })
      .listDirection(Axis.Vertical)
      .key('MyList2_swipe')
      .width('90%')

      Button('Collapse All')
        .width('60%')
        .height(80)
        .key('MyList2CloseButton')
        .onClick(() => {
          console.info('onClick closeAllSwipeActions')
          this.listScroller.closeAllSwipeActions({
            onFinish: () => {
              this.closeButtonContent = 'onFinish'
            }
          });
        })
      Text(this.list1Item0State)
        .key('swipe0100')
        .width('60%')
        .height(80)
        .backgroundColor(0x0000ff)
      Text(this.list2Item0State)
        .key('swipe0300')
        .width('60%')
        .height(80)
        .backgroundColor(0x0000ff)
      Text(this.closeButtonContent)
        .key('swipe0400')
        .width('60%')
        .height(80)
        .backgroundColor(0x0000ff)
      Text(this.list1Item1Offset)
        .key('swipe0500')
        .width('60%')
        .height(80)
        .backgroundColor(0x0000ff)
    }
    .padding(10)
    .backgroundColor(0xDCDCDC)
    .width('100%')
    .height('100%')
  }
}
