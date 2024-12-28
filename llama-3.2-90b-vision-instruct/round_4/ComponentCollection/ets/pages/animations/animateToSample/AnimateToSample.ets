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
  @State heightSize: number = 80;
  @State rotateAngle: number = 0;
  @State rotateAngleCurve: number = 0;
  @State rotateAngleDelay: number = 0;
  @State rotateAngleCount: number = 0;
  @State rotateAngleSpeed: number = 0;
  @State rotateAnglePlayback: number = 0;
  @State rotateAngleComplete: number = 0;
  @State flag: boolean = true;

  build() {
    Column() {
      TitleBar({ title: $r('app.string.animate_to') })
      Scroll() {
        Column() {
          Button($r('app.string.attribute_animation_change_size'))
            .onClick(() => {
              let originalHeight = this.heightSize;
              let targetHeight = originalHeight == 60 ? 80 : 60;
              animateTo({ duration: 1000 }, () => {
                this.heightSize = targetHeight;
              })
              this.flag = !this.flag;
            })
            .margin(35)
            .scale({ x: 1, y: this.heightSize / 80 })
            .width(160)
            .height(80)

          Button($r('app.string.attribute_animation_change_rotation'))
            .scale({ x: 1, y: 1 })
            .onClick(() => {
              animateTo({
                duration: 1200,
                curve: Curve.Friction,
                delay: 500,
                iterations: 2,
                playMode: PlayMode.Alternate,
                onFinish: () => {
                  this.rotateAngle = 0;
                }
              }, () => {
                let rotateAngle = 90;
                this.rotateAngle = rotateAngle;
              });
            })
            .width(160)
            .margin(35)
            .rotate({ angle: this.rotateAngle })

          Button($r('app.string.attribute_animation_set_curve'))
            .scale({ x: 1, y: 1 })
            .onClick(() => {
              animateTo({
                duration: 1200,
                curve: Curve.Sharp,
                delay: 500,
                iterations: 2,
                playMode: PlayMode.Alternate,
                onFinish: () => {
                  this.rotateAngleCurve = 0;
                }
              }, () => {
                let rotateAngleCurve = 90;
                this.rotateAngleCurve = rotateAngleCurve;
              });
            })
            .width(160)
            .rotate({ angle: this.rotateAngleCurve })
            .margin(35)

          Button($r('app.string.attribute_animation_delay'))
            .id('animate_btn_delay')
            .scale({ x: 1, y: 1 })
            .onClick(() => {
              animateTo({
                duration: 1200,
                curve: Curve.Friction,
                delay: 3000,
                iterations: 2,
                playMode: PlayMode.Alternate,
                onFinish: () => {
                  this.rotateAngleDelay = 0;
                }
              }, () => {
                let rotateAngleDelay = 90;
                this.rotateAngleDelay = rotateAngleDelay;
              });
            })
            .width(160)
            .rotate({ angle: this.rotateAngleDelay })
            .margin(35)

          Button($r('app.string.attribute_animation_play_count'))
            .scale({ x: 1, y: 1 })
            .onClick(() => {
              animateTo({
                duration: 1200,
                curve: Curve.Friction,
                delay: 500,
                iterations: 4,
                playMode: PlayMode.Alternate,
                onFinish: () => {
                  this.rotateAngleCount = 0;
                }
              }, () => {
                let rotateAngleCount = 90;
                this.rotateAngleCount = rotateAngleCount;
              });
            })
            .width(160)
            .rotate({ angle: this.rotateAngleCount })
            .margin(35)

          Button($r('app.string(attribute_animation_play_speed'))
            .scale({ x: 1, y: 1 })
            .onClick(() => {
              animateTo({
                duration: 1200,
                curve: Curve.Friction,
                delay: 500,
                iterations: 2,
                playMode: PlayMode.Alternate,
                tempo: 5,
                onFinish: () => {
                  this.rotateAngleSpeed = 0;
                }
              }, () => {
                let rotateAngleSpeed = 90;
                this.rotateAngleSpeed = rotateAngleSpeed;
              });
            })
            .width(160)
            .rotate({ angle: this.rotateAngleSpeed })
            .margin(35)

          Button($r('app.string.attribute_animation_unlimited_playback'))
            .scale({ x: 1, y: 1 })
            .onClick(() => {
              animateTo({
                duration: 1200,
                curve: Curve.Friction,
                delay: 500,
                iterations: -1, // -1为无限播放
                playMode: PlayMode.Alternate,
                tempo: 10
              }, () => {
                let rotateAnglePlayback = 90;
                this.rotateAnglePlayback = rotateAnglePlayback;
              });
            })
            .width(160)
            .rotate({ angle: this.rotateAnglePlayback })
            .margin(35)

          Button($r('app.string.attribute_animation_play_complete'))
            .scale({ x: 1, y: 1 })
            .onClick(() => {
              animateTo({
                duration: 1200,
                curve: Curve.Friction,
                delay: 500,
                iterations: 3,
                playMode: PlayMode.Alternate,
                tempo: 5,
                onFinish: () => {
                  this.rotateAngleComplete = 0;
                  ShowToast.shortToast($r('app.string.animate_over'));
                }
              }, () => {
                let rotateAngleComplete = 90;
                this.rotateAngleComplete = rotateAngleComplete;
              });
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