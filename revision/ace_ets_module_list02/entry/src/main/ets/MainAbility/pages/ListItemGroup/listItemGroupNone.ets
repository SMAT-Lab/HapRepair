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
struct ListItemGroupNoneExample {
  @State text:string = '1'
  private itemAreaTable: ItemAreaTable[] = [
    {
      title: '星期一',
      projects: ['语文', '数学']
    },
    {
      title: '星期二',
      projects: ['物理', '化学']
    }
  ]

  @Builder
  itemHead(text: string) {
    Text(text)
      .fontSize(20)
      .backgroundColor(0xAABBCC)
      .width('100%')
      .padding(10)
  }

  @Builder
  itemFoot(num: number) {
    Text('共' + num + '节课')
      .fontSize(16)
      .backgroundColor(0xAABBCC)
      .width('100%')
      .padding(5)
  }

  build() {
    Column({ space:20 }) {
      Text(this.text)//itemGroupArea: 0
        .id('itemGroupArea_none')
      List({ space: 20 }) {
        ForEach(this.itemAreaTable, (item: ItemAreaTable) => {
          ListItemGroup() {
          }
        })
      }
      .width('90%')
      .height('50%')
      .sticky(StickyStyle.None)
      .scrollBar(BarState.Off)
      .onScrollVisibleContentChange((start: VisibleListContentInfo, end: VisibleListContentInfo) => {
        if (start.itemGroupArea === 0) {
          //列表项组区域为无,ListItemGroup内部ListItem区域、header区域以及footer区域以外的区域
          this.text = 'ListItemGroupArea.NONE'
          console.info('ListItemGroupArea NONE success')
        } else if (start.itemGroupArea === 1) {  //ListItemGroup内部ListItem区域
          this.text = 'ListItemGroupArea.IN_LIST_ITEM_AREA'
          console.info('ListItemGroupArea IN_LIST_ITEM_AREA success')
        } else if (start.itemGroupArea === 2) {  //ListItemGroup内部header区域
          this.text = 'ListItemGroupArea.IN_HEADER_AREA'
          console.info('ListItemGroupArea IN_HEADER_AREA success')
        } else if (start.itemGroupArea === 3) {  //ListItemGroup内部footer区域
          this.text = 'ListItemGroupArea.IN_FOOTER_AREA'
          console.info('ListItemGroupArea IN_FOOTER_AREA success')
        }
        console.log(' start index: ' + start.index +
          ' start item group area: ' + start.itemGroupArea +
          ' start index in group: ' + start.itemIndexInGroup)
        console.log(' end index: ' + end.index +
          ' end item group area: ' + end.itemGroupArea +
          ' end index in group: ' + end.itemIndexInGroup)
      })
      .onDidScroll((scrollOffset: number, scrollState: ScrollState) => {
        console.info(`onScroll scrollState = ScrollState` + scrollState + `, scrollOffset = ` + scrollOffset)
      })
    }.width('100%').height('100%').backgroundColor(0xDCDCDC).padding({ top: 5 })
  }
}

interface ItemAreaTable {
  title: string;
  projects: string[];
}
