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
// ListItemGroup_constraintSize_设置了width、minWidth、maxWidth
@Entry
@Component
struct constraintSizeListItemGroup3 {
  private conTable3: CONTable3[] = [
    {
      title: '星期一',
      projects: ['语文', '数学', '英语']
    },
    {
      title: '星期二',
      projects: ['物理', '化学', '生物']
    },
    {
      title: '星期三',
      projects: ['历史', '地理', '政治']
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

  build() {
    Column() {
      List() {
        ForEach(this.conTable3, (item: CONTable3) => {
          ListItemGroup({ header: this.itemHead(item.title) }) {
            ForEach(item.projects, (project: string) => {
              ListItem() {
                Text(project)
                  .width('100%')
                  .height(100)
                  .fontSize(20)
                  .textAlign(TextAlign.Center)
                  .backgroundColor(0xFFFFFF)
              }
            }, (item: string) => item)
          }
          .id('group_constraintSize_3')
          .width(200)
          .constraintSize({minWidth:100, maxWidth:300})
          .divider({ strokeWidth: 1, color: Color.Blue }) // 每行之间的分界线
        })
      }
      .width('100%')
      .height('40%')

    }.width('100%')
    .height('100%')
    .backgroundColor(0xDCDCDC)
    .margin({ top: 5 })
  }
}


interface CONTable3 {
  title: string;
  projects: string[];
}