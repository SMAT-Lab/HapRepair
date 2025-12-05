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
struct comSizeAbilityTest_24 {
  // isLunar 未发生变化，改为普通变量
  isLunar: boolean = false
  private selectedDate: Date = new Date('2021-08-08')

  build() {
    Scroll() {
      Column({ space: 20 }) {
        Column() {
          DatePicker({
            start: new Date('1970-1-1'),
            end: new Date('2100-1-1'),
            selected: this.selectedDate
          })
            .disappearTextStyle({ color: Color.Gray, font: { size: '14fp', weight: FontWeight.Bold } })
            .textStyle({ color: '#ff182431', font: { size: '12fp', weight: FontWeight.Normal } })
            .selectedTextStyle({ color: '#ff0000FF', font: { size: '14fp', weight: FontWeight.Regular } })
            .lunar(this.isLunar)
            .backgroundColor(0XBDDB69)
            .key("DatePicker_Size")
            .size({ width: 120, height: 100 })
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
        }.key('DatePicker_Father')
      }.width('100%')
    }.height('100%')
  }
}