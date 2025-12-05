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

@Entry
@Component
struct ListDelete {
  @State arr: number[] = [0, 1, 2]
  @State arr2: number[] = [0, 1, 2]
  @State tSting: string = 'no action trigger'

  @Builder
  itemEnd(index: number, which: string) {
    Row() {
      Button('Del').margin('4vp')
        .onClick(() => {
          this.arr.splice(index, 1)
        }
        )
        .key(which + '_ListItem' + index + '_Button_Del')
      Button('Set').margin('4vp')
        .key(which + '_ListItem' + index + '_Button_Set')
    }.padding('4vp').justifyContent(FlexAlign.SpaceEvenly)
  }

  @Builder
  itemStart(index: number, which: string) {
    Row() {
      Button('Del').margin('4vp')
        .onClick(() => {
          this.arr.splice(index, 1)
        }
        )
        .key(which + '_ListItem' + index + '_Button_Del')
      Button('Set').margin('4vp')
        .key(which + '_ListItem' + index + '_Button_Set')
    }.padding('4vp').justifyContent(FlexAlign.SpaceEvenly)
  }

  build() {
    Column() {
      Text(this.tSting).key('MyList5ListItemOnDeleteH')
      List({ space: 2, initialIndex: 0 }) {
        ForEach(this.arr, (item: number, index) => {
          ListItem() {
            Text('' + item)
              .width('100%')
              .height(30)
              .fontSize(16)
              .textAlign(TextAlign.Center)
              .borderRadius(10)
              .backgroundColor(0xFFFFFF)
          }.swipeAction({ end: (): void => this.itemEnd( index, 'MyList1'), edgeEffect: SwipeEdgeEffect.None })
          .key('MyList1_ListItem' + index)
        }, (item: string) => item + '')
      }
      .margin({ top: 5 })
      .divider({ strokeWidth: 2, color: 0xFFFFFF, startMargin: 20, endMargin: 20 })
      .width('90%')
      .height('15%')
      .key('MyList1_del')

      List({ space: 2, initialIndex: 0 }) {
        ForEach(this.arr, (item: number, index) => {
          ListItem() {
            Text('' + item)
              .width('100%')
              .height(30)
              .fontSize(16)
              .textAlign(TextAlign.Center)
              .borderRadius(10)
              .backgroundColor(0xFFFFFF)
          }
          .swipeAction({ start: (): void => this.itemStart( index, 'MyList2'), edgeEffect: SwipeEdgeEffect.None })
          .key('MyList2_ListItem' + index)
        }, (item: string) => item + '')
      }
      .margin({ top: 5 })
      .divider({ strokeWidth: 2, color: 0xFFFFFF, startMargin: 20, endMargin: 20 })
      .width('90%')
      .height('15%')
      .key('MyList2_del')

      List({ space: 2, initialIndex: 0 }) {
        ForEach(this.arr, (item: number, index) => {
          ListItem() {
            Text('' + item)
              .width('100%')
              .height(30)
              .fontSize(16)
              .textAlign(TextAlign.Center)
              .borderRadius(10)
              .backgroundColor(0xFFFFFF)
          }.swipeAction({ end: (): void => this.itemEnd( index, 'MyList3'), edgeEffect: SwipeEdgeEffect.Spring })
          .key('MyList3_ListItem' + index)
        }, (item: string) => item + '')
      }
      .margin({ top: 5 })
      .divider({ strokeWidth: 2, color: 0xFFFFFF, startMargin: 20, endMargin: 20 })
      .width('90%')
      .height('15%')
      .key('MyList3_del')

      List({ space: 2, initialIndex: 0 }) {
        ForEach(this.arr, (item: number, index) => {
          ListItem() {
            Text('' + item)
              .width('100%')
              .height(30)
              .fontSize(16)
              .textAlign(TextAlign.Center)
              .borderRadius(10)
              .backgroundColor(0xFFFFFF)
          }.swipeAction({ start: (): void => this.itemStart( index, 'MyList4'), edgeEffect: SwipeEdgeEffect.Spring })
          .key('MyList4_ListItem' + index)
        }, (item: string) => item + '')
      }
      .margin({ top: 5 })
      .divider({ strokeWidth: 2, color: 0xFFFFFF, startMargin: 20, endMargin: 20 })
      .width('90%')
      .height('15%')
      .key('MyList4_del')

      List({ space: 2, initialIndex: 0 }) {
        ForEach(this.arr2, (item: number, index) => {
          ListItem() {
            Text('' + item)
              .width('100%')
              .height(30)
              .fontSize(16)
              .textAlign(TextAlign.Center)
              .borderRadius(10)
              .backgroundColor(0xFFFFFF)
          }
          .swipeAction({ end: {
            builder: (): void => this.itemEnd( index, 'MyList5'),
            onAction: () => {
              console.info('MyList5_ListItem' + index + ' onAction')
              this.arr2.splice(index, 1)
              this.tSting = 'action trigger';

            },
            actionAreaDistance: 5,
            onEnterActionArea: () => {
            },
            onExitActionArea: () => {
            }
          }
          })
          .key('MyList5_ListItem' + index)
        }, (item: string) => item + '')
      }
      .height('15%')
      .margin({ top: 5 })
      .divider({ strokeWidth: 2, color: 0xFFFFFF, startMargin: 20, endMargin: 20 })
      .width('90%')
      .key('MyList5_del')

    }
    .width('100%')
    .backgroundColor(0xDCDCDC)
  }
}
