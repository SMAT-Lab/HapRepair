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
export struct TabsView {
  // 直接使用一般变量即可
  fontColor: string = '#182431'
  selectedFontColor: string = '#007DFF'
  @State currentIndex: number = 0
  private controller: TabsController = new TabsController()
  @Link _position: Position;
  private componentKey: string;

  @Builder
  TabBuilder(index: number, name: string) {
    Column() {
      Text(name)
        .fontColor(this.currentIndex === index ? this.selectedFontColor : this.fontColor)
        .fontSize(16)
        .fontWeight(this.currentIndex === index ? 500 : 400)
        .lineHeight(22)
        .margin({ top: 17, bottom: 7 })
      Divider()
        .strokeWidth(2)
        .color('#007DFF')
        .opacity(this.currentIndex === index ? 1 : 0)
    }.width('100%')
  }

  build() {
    Tabs({ barPosition: BarPosition.Start, controller: this.controller }) {
      TabContent() {
        Column().width('100%').height('100%').backgroundColor('#00CB87')
      }.tabBar(this.TabBuilder(0, 'green'))

      TabContent() {
        Column().width('100%').height('100%').backgroundColor('#007DFF')
      }.tabBar(this.TabBuilder(1, 'blue'))

      TabContent() {
        Column().width('100%').height('100%').backgroundColor('#FFBF00')
      }.tabBar(this.TabBuilder(2, 'yellow'))

      TabContent() {
        Column().width('100%').height('100%').backgroundColor('#E67C92')
      }.tabBar(this.TabBuilder(3, 'pink'))
    }
    .vertical(false)
    .barMode(BarMode.Fixed)
    .barWidth(360)
    .barHeight(56)
    .animationDuration(400)
    .onChange((index: number) => {
      this.currentIndex = index
    })
    .width(360)
    .height(296)
    .backgroundColor('#F1F3F5')
    .position(this._position)
    .key(this.componentKey)
  }
}