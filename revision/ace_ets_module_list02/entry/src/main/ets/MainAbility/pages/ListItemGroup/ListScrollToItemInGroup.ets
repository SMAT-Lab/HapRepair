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
class WordGroup {
  title: string;
  words: string[];
  constructor(title: string, words: string[]) {
    this.title = title;
    this.words = words;
  }
}

@Entry
@Component
struct WordListSample {
  wordsGroup: WordGroup[] = [
    new WordGroup('A', ['Apple', 'App', 'An', 'Ace']),
    new WordGroup('B', ['Banana', 'Blackberry', 'Blue']),
    new WordGroup('C', ['Cake', 'Chicken', 'Class']),
    new WordGroup('D', ['Dark', 'Dress', 'Dance', 'Dear', 'Date', 'Department', 'Dog', 'Damage', 'Decade',
      'Drum', 'Daisy', 'Deity'])
  ]

  illegalGroupNo:number[] = [this.wordsGroup.length + 1, 0];
  illegalIndexNo:number[] = [0, -5];

  scrollAlignStyle: ScrollAlign[] = [
    ScrollAlign.START,
    ScrollAlign.CENTER,
    ScrollAlign.END,
    ScrollAlign.AUTO
  ]

  listScroller: ListScroller = new ListScroller()

  @Builder itemHead(text: string) {
    Text(text)
      .fontSize(20)
      .backgroundColor('#fff1f3f5')
      .width('100%')
      .padding(5)
  }

  @Builder
  itemFoot(num: number) {
    Text('ItemCount:' + num)
      .fontSize(16)
      .backgroundColor(0xAABBCC)
      .width('100%')
      .padding(5)
  }

  build() {
    Column({ space: 10 }) {
      List({ space: 5}) {
        ForEach(this.wordsGroup, (item: WordGroup) => {
          ForEach(item.words, (word: string) => {
            ListItem({ style: ListItemStyle.NONE }) {
              Button('Item_' + item.title + '_' + word)
                .backgroundColor(Color.Green)
                .onClick(() => {
                  console.info('WordListSample group:' + this.wordsGroup.indexOf(item) + ' ' +
                    'index:' + item.words.indexOf(word))
                  this.listScroller.scrollToItemInGroup(
                    this.wordsGroup.indexOf(item),
                    item.words.indexOf(word), true,
                    this.scrollAlignStyle[this.wordsGroup.indexOf(item)])
                })
                .key('Btn_' + this.wordsGroup.indexOf(item) + '_' + item.words.indexOf(word))
            }
          })
        })
        // group not exist
        ListItem({ style: ListItemStyle.NONE }) {
          Button('Item_GroupNo_Not' + '_' + 'Exsit')
            .backgroundColor(Color.Red)
            .onClick(() => {
              this.listScroller.scrollToItemInGroup(
                this.illegalGroupNo[0],
                this.illegalGroupNo[1], true,
                ScrollAlign.AUTO)
            })
            .key('Btn_' + 'Err_' + 'Group')
        }
        // index not exist
        ListItem({ style: ListItemStyle.NONE }) {
          Button('Item_IndexNo_Not' + '_' + 'Exsit')
            .backgroundColor(Color.Red)
            .onClick(() => {
              this.listScroller.scrollToItemInGroup(
                this.illegalIndexNo[0],
                this.illegalIndexNo[1], true,
                ScrollAlign.AUTO)
            })
            .key('Btn_' + 'Err_' + 'Index')
        }
      }.lanes(4)

      Text('words-table')
        .textAlign(TextAlign.Start)
        .width('100%')
        .fontSize(25)
        .backgroundColor('#fff1f3f5')
        .borderRadius(10)

      Stack() {
        List({ initialIndex: 0, scroller: this.listScroller }) {
          ForEach(this.wordsGroup, (item: WordGroup) => {
            ListItemGroup({ header: this.itemHead(item.title),
              footer: this.itemFoot(item.words.length), style: ListItemGroupStyle.NONE }) {
              // 循环渲染ListItem
              ForEach(item.words, (word: string) => {
                ListItem({ style: ListItemStyle.NONE }) {
                  Text(word)
                    .textAlign(TextAlign.Start)
                    .width('80%')
                    .fontSize(25)
                    .backgroundColor('#fff1f3f5')
                }
                .height('10%')
                .width('100%')
                .backgroundColor(Color.Green)
                .key('ListItem_Group' + this.wordsGroup.indexOf(item) + '_Index' + item.words.indexOf(word))
              })
            }
          })
        }
        .key('MyList1')
        .scrollBar(BarState.On)
        .divider({
          strokeWidth: 4,
          color: '#ff0000ff'
        })
        Divider().strokeWidth(1).backgroundColor(Color.Red).key('CenterLine')
      }
    }
    .width('100%').height('100%').backgroundColor(0xDCDCDC).padding({ top: 5 })
  }
}