
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
struct comSizeAbilityTest_23 {
  @State state: AnimationStatus = AnimationStatus.Initial
  // reverse 未发生变化，改为普通变量
  reverse: boolean = false
  // iterations 未发生变化，改为普通变量
  iterations: number = 1

  build() {
    Scroll() {
      Column({ space: 20 }) {
        Column() {
          ImageAnimator()
            .size({ width: 340, height: 240 })
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
            .key("ImageAnimator_Size")
            .images([
              {
                src: $r('app.media.icon'), top: 100
              },
              {
                src: $r('app.media.icon')
              },
              {
                src: $r('app.media.icon')
              },
              {
                src: $r('app.media.icon')
              }
            ])
            .duration(2000)
            .state(this.state)
            .reverse(this.reverse)
            .fillMode(FillMode.None)
            .iterations(this.iterations)
            .width(340)
            .height(240)
            .margin({ top: 100 })
            .onStart(() => {
              console.info('Start')
            })
            .onPause(() => {
              console.info('Pause')
            })
            .onFinish(() => {
              console.info('Finish')
              this.state = AnimationStatus.Stopped
            })
          Row() {
            Button('start').width(100).padding(5).onClick(() => {
              this.state = AnimationStatus.Running
            }).margin(5)
            Button('pause').width(100).padding(5).onClick(() => {
              this.state = AnimationStatus.Paused // 显示当前帧图片
            }).margin(5)
            Button('stop').width(100).padding(5).onClick(() => {
              this.state = AnimationStatus.Stopped // 显示动画的起始帧图片
            }).margin(5)
          }
        }.key('ImageAnimator_Father')
      }.width('100%')
    }.height('100%')
  }
}

