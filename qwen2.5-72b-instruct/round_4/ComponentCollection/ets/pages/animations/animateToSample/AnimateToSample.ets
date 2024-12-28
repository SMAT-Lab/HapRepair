

/*
 * Copyright (c) 2023 Hunan OpenValley Digital Industry Development Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import { TitleBar } from '../../../common/TitleBar';
import ShowToast from '../../../util/ShowToast';

@Entry
@Component
struct AnimationToSample {
  heightSize: number = 80;
  build() {
    Column() {
      TitleBar({ title: $r('app.string.animate_to') })
      Scroll() {
        Column() {
          Button($r('app.string.attribute_animation_change_size'))
            .onClick(() => {
              if (this.flag) {
                animateTo({ duration: 2000, curve: Curve.EaseOut, iterations: 1, playMode: PlayMode.Normal, onFinish: () => ShowToast.shortToast($r('app.string.pan_end')) })
                this.scale({ x: 0.75, y: 0.75 });
              } else {
                animateTo({})
                this.scale({ x: 1, y: 1 });
              }
              this.flag = !this.flag;
            })
            .margin(35)
            .width(160)
            .height(this.heightSize)

          Button($r('app.string.attribute_animation_change_rotation'))
            .onClick(() => {
              animateTo({ duration: 1200, curve: Curve.Friction, delay: 500, iterations: 2, playMode: PlayMode.Alternate, onFinish: () => this.rotateAngle = 0 })
              this.rotate({ angle: 90 });
            })
            .width(160)
            .margin(35)
            .rotate({ angle: this.rotateAngle })

          Button($r('app.string.attribute_animation_set_curve'))
            .onClick(() => {
              animateTo({ duration: 1200, curve: Curve.Sharp, delay: 500, iterations: 2, playMode: PlayMode.Alternate, onFinish: () => this.rotateAngleCurve = 0 })
              this.rotate({ angle: 90 });
            })
            .width(160)
            .rotate({ angle: this.rotateAngleCurve })
            .margin(35)

          Button($r('app.string.attribute_animation_delay'))
            .id('animate_btn_delay')
            .onClick(() => {
              animateTo({ duration: 1200, curve: Curve.Friction, delay: 3000, iterations: 2, playMode: PlayMode.Alternate, onFinish: () => this.rotateAngleDelay = 0 })
              this.rotate({ angle: 90 });
            })
            .width(160)
            .rotate({ angle: this.rotateAngleDelay })
            .margin(35)

          Button($r('app.string.attribute_animation_play_count'))
            .onClick(() => {
              animateTo({ duration: 1200, curve: Curve.Friction, delay: 500, iterations: 4, playMode: PlayMode.Alternate, onFinish: () => this.rotateAngleCount = 0 })
              this.rotate({ angle: 90 });
            })
            .width(160)
            .rotate({ angle: this.rotateAngleCount })
            .margin(35)

          Button($r('app.string.attribute_animation_play_speed'))
            .onClick(() => {
              animateTo({ duration: 1200, curve: Curve.Friction, delay: 500, iterations: 2, playMode: PlayMode.Alternate, tempo: 5, onFinish: () => this.rotateAngleSpeed = 0 })
              this.rotate({ angle: 90 });
            })
            .width(160)
            .rotate({ angle: this.rotateAngleSpeed })
            .margin(35)

          Button($r('app.string.attribute_animation_unlimited_playback'))
            .onClick(() => {
              animateTo({ duration: 1200, curve: Curve.Friction, delay: 500, iterations: -1, playMode: PlayMode.Alternate, tempo: 10 })
              this.rotate({ angle: 90 });
            })
            .width(160)
            .rotate({ angle: this.rotateAnglePlayback })
            .margin(35)

          Button($r('app.string.attribute_animation_play_complete'))
            .onClick(() => {
              animateTo({ duration: 1200, curve: Curve.Friction, delay: 500, iterations: 3, playMode: PlayMode.Alternate, tempo: 5, onFinish: () => { this.rotateAngleComplete = 0; ShowToast.shortToast($r('app.string.animate_over')) } })
              this.rotate({ angle: 90 });
            })
            .width(160)
            .rotate({ angle: this.rotateAngleComplete })
            .margin(50)
        }
        .constraintSize({ minHeight: '100%' })
      }
      .borderRadius(24)
      .backgroundColor(Color.White)
      .height('90%')
      .width('100%')
      .margin({ left: 12, right: 12 })
    }
    .height('100%')
    .width('100%')
    .backgroundColor($r('app.color.background_shallow_grey'))
  }
}