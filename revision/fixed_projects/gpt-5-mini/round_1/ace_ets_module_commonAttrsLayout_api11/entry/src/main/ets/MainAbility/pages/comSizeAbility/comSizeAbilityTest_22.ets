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
struct comSizeAbilityTest_22 {
  build() {
    Scroll() {
      Column({ space: 20 }) {
        Column() {
          Checkbox({ name: 'checkbox1', group: 'checkboxGroup' })
            .select(true)
            .selectedColor(0xed6f21)
            .backgroundColor(0XBDDB69)
            .key("Checkbox_Size")
            .size({ width: 100, height: 60 })
          DataPanel({ values: [0] })
            .trackShadow({ offsetX: 5, offsetY: 5 })
            .backgroundColor('black')
            .width(200)
            .height(200)
            .key("DataPanel_Offset")
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
        }.backgroundColor('red').key('DataPanel_Father')
      }.width('100%')
    }.height('100%')
  }
}


