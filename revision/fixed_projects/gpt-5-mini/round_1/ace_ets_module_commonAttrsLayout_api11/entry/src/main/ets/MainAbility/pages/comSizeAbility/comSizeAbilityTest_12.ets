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
struct comSizeAbilityTest_12 {
  // isRefreshing 未发生变化，改为普通变量
  isRefreshing: boolean = false
  // counter 未发生变化，改为普通变量
  counter: number = 0

  build() {
    Scroll() {
      Column({ space: 20 }) {
        Column() {
          Rect()
            .fill(Color.Pink)
            .key("Rect_Size")
            .size({ width: 210, height: 130 })
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
        }.key('Rect_Father')

        Column() {
          Refresh({ refreshing: $$this.isRefreshing, offset: 120, friction: 100 }) {
            Text('下拉刷新: ' + this.counter)
              .fontSize(30)
              .margin(10)
          }
          .backgroundColor('gray')
          .key("Refresh_Size")
          .size({ width: 220, height: 140 })
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
        }.key('Refresh_Father')
      }.width('100%')
    }.height('100%')
  }
}