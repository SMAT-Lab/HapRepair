/*
 * Copyright (c) 2023 Huawei Device Co., Ltd.
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

@Component
export struct ListItemGroupView {
  @Link _markAnchor: Position;
  private componentKey: string;
  private parentWidth: number;
  private parentHeight: number;
  private subassemblyWidth: number;
  private subassemblyHeight: number;
  private projects: string[] = ['Chinese', 'mathematics', 'English']

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
    Text('have ' + num + 'classes')
      .fontSize(16)
      .backgroundColor(0xAABBCC)
      .width('100%')
      .padding(5)
  }

  @Builder
  content(project: string) {
    Text(project)
      .width('100%')
      .height(100)
      .fontSize(20)
      .textAlign(TextAlign.Center)
      .backgroundColor(0xFFFFFF)
  }

  build() {
    List({ space: 20 }) {
      ListItemGroup({ header: this.itemHead('Monday'), footer: this.itemFoot(this.projects.length) }) {
        ForEach(this.projects, (project: string) => {
          ListItem() {
            this.content(project)
          }
        }, (item: string) => item)
      }
      .width(this.subassemblyWidth)
      .height(this.subassemblyHeight)
      .borderRadius(20)
      .markAnchor(this._markAnchor)
      .key(this.componentKey)
      .divider({ strokeWidth: 1, color: Color.Blue }) // line of demarcation

      ListItemGroup({ header: this.itemHead('Tuesday'), footer: this.itemFoot(this.projects.length) }) {
        ForEach(this.projects, (project: string) => {
          ListItem() {
            this.content(project)
          }
        }, (item: string) => item)
      }
      .borderRadius(20)
      .divider({ strokeWidth: 1, color: Color.Blue }) // line of demarcation
    }
    .width(this.parentWidth)
    .height(this.parentHeight)
    .sticky(StickyStyle.Header | StickyStyle.Footer)
  }
}