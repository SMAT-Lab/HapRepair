

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
struct comSizeAbilityTest_01 {
  private settings: RenderingContextSettings = new RenderingContextSettings(true)
  private context: CanvasRenderingContext2D = new CanvasRenderingContext2D(this.settings)
  scroller: Scroller = new Scroller()

  build() {
    Column() {
      Row() {
        Button('to_1').onClick(()=>{
          this.scroller.scrollToIndex(0)
        }).key('comSizeAbilityTest_01_to_1')

        Button('to_2').onClick(()=>{
          this.scroller.scrollToIndex(1)
        }).key('comSizeAbilityTest_01_to_2')

        Button('to_3').onClick(()=>{
          this.scroller.scrollToIndex(2)
        }).key('comSizeAbilityTest_01_to_3')
      }

      List({ space: 5, initialIndex: 0, scroller: this.scroller }) {
        ListItem() {
          Badge({
            count: 1,
            position: BadgePosition.Left,
            style: { badgeSize: 20, badgeColor: '#FA2A2D' }
          }) {
            Text('文字部分').width(100).height(50).backgroundColor('blue')
          }
          .key("Badge_Size")
          .size({ width: 170, height: 120 })
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
        }

        ListItem() {
          Button()
            .key("Button_Size")
            .size({ width: 90, height: 90 })
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
        }

        ListItem() {
          Canvas(this.context)
            .width('100%')
            .height('100%')
            .backgroundColor('#ffff00')
            .onReady(() => {
              this.context.fillRect(0, 30, 100, 100)
            })
            .key("Canvas_Size")
            .size({ width: 211, height: 140 })
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
        }
      }.width('100%')
    }.height('100%')
  }
}