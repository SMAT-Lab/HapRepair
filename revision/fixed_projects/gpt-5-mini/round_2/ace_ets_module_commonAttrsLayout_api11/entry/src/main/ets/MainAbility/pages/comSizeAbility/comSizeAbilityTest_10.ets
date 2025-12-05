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
struct comSizeAbilityTest_10 {
  scroller: Scroller = new Scroller()
  build() {
    Column() {
      Row() {
        Button('to_1').onClick(()=>{
          this.scroller.scrollToIndex(0)
        }).key('comSizeAbilityTest_10_to_1')

        Button('to_2').onClick(()=>{
          this.scroller.scrollToIndex(1)
        }).key('comSizeAbilityTest_10_to_2')

        Button('to_3').onClick(()=>{
          this.scroller.scrollToIndex(2)
        }).key('comSizeAbilityTest_10_to_3')
      }

      List({ space: 5, initialIndex: 0, scroller: this.scroller }) {
        ListItem() {
          Path()
            .commands('M0 0 H200 V200 H0 Z')// 单位转化
            .stroke(Color.Green)
            .strokeWidth(3)
            .key("Path_Size")
            .backgroundColor('red')
            .size({ width: 210, height: 132 })
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
        }.key('Path_Father')

        ListItem() {
          Polygon()
            .points([[0, 0], [50, 100], [100, 0]])
            .fill(Color.Green)
            .key("Polygon_Size")
            .backgroundColor('red')
            .size({ width: 210, height: 140 })
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
        }.key('Polygon_Father')

        ListItem() {
          Polyline()
            .fillOpacity(0)
            .stroke(Color.Green)
            .strokeWidth(8)
            .points([[20, 0], [0, 100], [100, 90]])
            .strokeLineJoin(LineJoinStyle.Round)
            .strokeLineCap(LineCapStyle.Round)
            .backgroundColor('red')
            .key("Polyline_Size")
            .size({ width: 210, height: 125 })
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
        }.key('Polyline_Father')
      }.width('100%')
    }.height('100%')
  }
}