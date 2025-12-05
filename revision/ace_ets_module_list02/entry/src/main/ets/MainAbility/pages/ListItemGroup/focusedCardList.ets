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
struct focusedCardListItem{
  scroller: Scroller = new Scroller()

  aboutToAppear() {
    console.info('listItem_focused about to appear')
  }

  @Builder
  itemHead(text: string) {
    Text(text)
      .fontSize(20)
      .backgroundColor(0xAABBCC)
      .width('100%')
      .padding(10)
  }

  build() {
    Column() {
      List({ space: '4vp', initialIndex: 0, scroller: this.scroller }) {
        ListItemGroup({ header: this.itemHead('header'), style: ListItemGroupStyle.CARD }) {
          ListItem({ style: ListItemStyle.CARD }) {
            Text('ListItemGroupStyle.CARD与ListItemStyle.CARD')
              .id('listItem_focused_card')
              .width('100%')
              .height(60)
              .focusable(true)
              .textAlign(TextAlign.Center)
              .backgroundColor(Color.Pink)
          }
        }

      }
      .width('90%')
      .height('50%')
    }
    .width('100%')
    .height('100%')
    .margin({ top: 10 })
  }
}