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
struct comSizeAbilityTest_03 {
  scroller: Scroller = new Scroller()
  build() {
    Column() {
      Row() {
        Button('to_1').onClick(()=>{
          this.scroller.scrollToIndex(0)
        }).key('comSizeAbilityTest_03_to_1').width('25%')

        Button('to_2').onClick(()=>{
          this.scroller.scrollToIndex(1)
        }).key('comSizeAbilityTest_03_to_2').width('25%')

        Button('to_3').onClick(()=>{
          this.scroller.scrollToIndex(2)
        }).key('comSizeAbilityTest_03_to_3').width('25%')

        Button('to_4').onClick(()=>{
          this.scroller.scrollToIndex(3)
        }).key('comSizeAbilityTest_03_to_4').width('25%')
      }

      List({ space: 5, initialIndex: 0, scroller: this.scroller }) {
        ListItem() {
          Circle()
            .backgroundColor('red')
            .key("Circle_Size")
            .size({ width: 310, height: 140 })

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
          Column()
            .backgroundColor('red')
            .key("Column_Size")
            .size({ width: 160, height: 100 })
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
          Counter()
            .key("Counter_Size")
            .size({ width: 150, height: 100 })
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
          Divider()
            .backgroundColor('red')
            .key("Divider_Size")
            .size({ width: 130, height: 100 })
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