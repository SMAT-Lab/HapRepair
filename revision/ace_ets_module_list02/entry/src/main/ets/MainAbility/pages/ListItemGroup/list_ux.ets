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

import Curves from '@ohos.curves';

@Entry
@Component
struct listUx{
  private scrollerForList: Scroller = new Scroller()
  @State message: string = 'Hello World'

  aboutToAppear() {
      console.info('listux about to appear')
  }

  build() {
    Row() {
      Column() {
        List({ space: '4vp', initialIndex: 0, scroller: this.scrollerForList }) {
          ListItemGroup() {
          }
          .key('nurmal')
          .margin({ left: $r('sys.float.ohos_id_card_margin_start'), right: $r('sys.float.ohos_id_card_margin_end') })

          ListItemGroup({ style: ListItemGroupStyle.CARD }) {
            ListItem({ style: ListItemStyle.CARD }) {
              Text('ListItemGroupStyle.CARD与ListItemStyle.CARD')
                .fontSize(10)
                .width('100%')
                .textAlign(TextAlign.Center)
            }
            .key('ArkUXII_List_style_0100')

            ListItem({ style: ListItemStyle.NONE }) {
              Text('ListItemGroupStyle.CARD与ListItemStyle.NONE')
                .fontSize(10)
                .width('100%')
                .textAlign(TextAlign.Center)
            }
            .key('ArkUXII_List_style_0300')

            ListItem({ style: null! }) {
              Text('ListItemGroupStyle.CARD与null')
                .fontSize(10)
                .width('100%')
                .textAlign(TextAlign.Center)
            }
            .key('ArkUXII_List_style_0400')

            ListItem({ style: undefined! }) {
              Text('ListItemGroupStyle.CARD与undefined')
                .fontSize(10)
                .width('100%')
                .textAlign(TextAlign.Center)
            }
            .key('ArkUXII_List_style_0500')

            ListItem({ style: ListItemStyle.NONE }) {
              Text('ListItemGroupStyle.CARD与\'\'')
                .fontSize(10)
                .width('100%')
                .textAlign(TextAlign.Center)
            }
            .key('ArkUXII_List_style_0600')

            ListItem() {
              Text('ListItemGroupStyle.CARD')
                .fontSize(10)
                .width('100%')
                .textAlign(TextAlign.Center)
            }
            .key('ArkUXII_List_style_0700')
          }
          .key('ListItemGroup')

          ListItemGroup({ style: ListItemGroupStyle.NONE }) {
            ListItem({ style: ListItemStyle.CARD }) {
              Text('ListItemGroupStyle.NONE与ListItemStyle.CARD')
                .fontSize(10)
                .width('100%')
                .textAlign(TextAlign.Center)
            }
          }
          .key('ArkUXII_List_style_0800')

          ListItemGroup({ style: null! }) {
            ListItem({ style: ListItemStyle.CARD }) {
              Text('null与ListItemStyle.CARD')
                .fontSize(10)
                .width('100%')
                .textAlign(TextAlign.Center)
            }
          }
          .key('ArkUXII_List_style_0900')

          ListItemGroup({ style: undefined! }) {
            ListItem({ style: ListItemStyle.CARD }) {
              Text('undefined与ListItemStyle.CARD')
                .fontSize(10)
                .width('100%')
                .textAlign(TextAlign.Center)
            }
          }
          .key('ArkUXII_List_style_1000')

          ListItemGroup({ style: ListItemGroupStyle.NONE }) {
            ListItem({ style: ListItemStyle.CARD }) {
              Text('\'\'与ListItemStyle.CARD')
                .fontSize(10)
                .width('100%')
                .textAlign(TextAlign.Center)
            }
          }
          .key('ArkUXII_List_style_1100')

          ListItemGroup() {
            ListItem({ style: ListItemStyle.CARD }) {
              Text('ListItemStyle.CARD')
                .fontSize(10)
                .width('100%')
                .textAlign(TextAlign.Center)
            }
          }
          .key('ArkUXII_List_style_1200')

          ListItemGroup({ style: ListItemGroupStyle.CARD }) {
            ListItem({ style: ListItemStyle.CARD }) {
              Text('\'\'之后再ListItemStyle.CARD与\'\'之后再ListItemGroupStyle.CARD')
                .fontSize(10)
                .width('100%')
                .textAlign(TextAlign.Center)
            }
            .key('ListItem_ArkUXII_List_style_0200')
          }
          .key('ArkUXII_List_style_0200')

          ListItem() {
            Button('scrollTo自定义弹簧曲线测试(Curves.springCurve(7, 1, 227, 33))')
              .key('scrollerForList')
              .onClick(() => {
                let curve = Curves.springCurve(7, 1, 227, 33)
                this.scrollerForList.scrollTo({
                  xOffset: 0,
                  yOffset: this.scrollerForList.currentOffset().yOffset + 600,
                  animation: true
                })
              })
              .height(45)
              .width('80%')
              .fontSize(13)
          }
        }
        .key('RollBack')
        .multiSelectable(true)

        List() {
        }
        .key('ArkUX_Stage_List_LimitAnimation_0100')
        .scrollSnapAlign(ScrollSnapAlign.NONE)

        List() {
        }
        .key('ArkUX_Stage_List_LimitAnimation_0200')
        .scrollSnapAlign(ScrollSnapAlign.START)

        List() {
        }
        .key('ArkUX_Stage_List_LimitAnimation_0300')
        .scrollSnapAlign(ScrollSnapAlign.CENTER)

        List() {
        }
        .key('ArkUX_Stage_List_LimitAnimation_0400')
        .scrollSnapAlign(ScrollSnapAlign.END)

        List() {
        }
        .key('ArkUX_Stage_List_LimitAnimation_0500')
        .scrollSnapAlign(null)

        List() {
        }
        .key('ArkUX_Stage_List_LimitAnimation_0600')
        .scrollSnapAlign(ScrollSnapAlign[''])

        List() {
        }
        .key('ArkUX_Stage_List_LimitAnimation_0700')
        .scrollSnapAlign(undefined)

        List() {
        }
        .key('ArkUX_Stage_List_LimitAnimation_0800')

        List() {
        }
        .key('ArkUX_Stage_List_LimitAnimation_0900')
        .scrollSnapAlign(ScrollSnapAlign[''])

        List() {
        }
        .key('ArkUX_Stage_List_LimitAnimation_1000')
        .scrollSnapAlign(ScrollSnapAlign.START)
        .scrollSnapAlign(ScrollSnapAlign.NONE)

        List() {
        }
        .key('ArkUX_Stage_List_LimitAnimation_1100')
        .scrollSnapAlign(ScrollSnapAlign.NONE)
        .scrollSnapAlign(ScrollSnapAlign.START)

        List() {
        }
        .key('ArkUX_Stage_List_LimitAnimation_1200')
        .scrollSnapAlign(ScrollSnapAlign.START)
        .scrollSnapAlign(ScrollSnapAlign[''])

        List() {
        }
        .key('ArkUX_Stage_List_LimitAnimation_1300')
        .scrollSnapAlign(ScrollSnapAlign[''])
        .scrollSnapAlign(ScrollSnapAlign.START)

        List() {
        }
        .key('ArkUX_Stage_List_LimitAnimation_1400')
        .scrollSnapAlign(0)

        List() {
        }
        .key('ArkUX_Stage_List_LimitAnimation_1500')
        .scrollSnapAlign(1)

        List() {
        }
        .key('ArkUX_Stage_List_LimitAnimation_1600')
        .scrollSnapAlign(6)
      }
      .width('100%')
    }
    .height('100%')
  }
}