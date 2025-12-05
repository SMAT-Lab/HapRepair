

/**
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
import matrix4 from '@ohos.matrix4'

let matrix = matrix4.init([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0])

@Entry
@Component
struct comSizeAbilityTest_15 {
  // fontColor 未发生变化，改为普通变量
  fontColor: string = '#182431'
  // selectedFontColor 未发生变化，改为普通变量
  selectedFontColor: string = '#007DFF'
  // currentIndex 未发生变化，改为普通变量
  currentIndex: number = 0
  private controller: TabsController = new TabsController()

  @Builder
  TabBuilder(index: number) {
    Column() {
      Text(`Tab${index + 1}`)
        .fontColor(this.currentIndex === index ? this.selectedFontColor : this.fontColor)
        .fontSize(10)
        .fontWeight(500)
        .lineHeight(14)
    }.width('100%')
  }

  build() {
    Scroll() {
      Column({ space: 20 }) {
          Tabs({ barPosition: BarPosition.End, controller: this.controller }) {
            TabContent() {
              Column() {
                Text('Tab1')
                  .fontSize(36)
                  .fontColor('#182431')
                  .fontWeight(500)
                  .opacity(0.4)
                  .margin({ top: 30, bottom: 56.5 })
              }
            }
            .key("TabContent_Size")
            .translate({ x: 1, y: 1, z: 1 })
            .scale({
              x: 1,
              y: 1,
              z: 1,
              centerX: 1,
              centerY: 1
            })
            .rotate({
              x: 1,
              y: 1,
              z: 1,
              angle: 0,
              centerX: 1,
              centerY: 1
            })
            .transform(matrix)
            .tabBar(this.TabBuilder(0))
            .backgroundColor(0XBDDB69)
          }
          .vertical(false)
          .barHeight(56)
          .height(196)
          .width(230)
          .backgroundColor('#F1F3F5')
          .margin({ top: 38 })
          .key('TabContent_Father')
        }
      }.width('100%')
    }.height('100%')
  }
}