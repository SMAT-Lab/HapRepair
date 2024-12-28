
```arkts


/*
 * Copyright (c) 2022 Huawei Device Co., Ltd.
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

import { NavigationBar } from "../../../common/components/navigationBar"
import { GetColor } from "../../../common/components/getColor"

@Entry
@Component
struct ProgressSample {
  @State setValue: number = 10;
  @State setProgressColor: string = 'rgba(0,0,0,1)'
  @State setProgressStyle: number  = ProgressStyle.Linear;
  @State progressStyleString: string = 'Linear';
  @State setSliderStep: number = 1;
  @State setSliderStyle: SliderStyle = SliderStyle.OutSet;

  build() {
    Flex({ direction: FlexDirection.Column, justifyContent: FlexAlign.Start, alignItems: ItemAlign.Center }) {
      NavigationBar({ title: 'Progress' })
      Column() {
        Flex({ direction: FlexDirection.Column, justifyContent: FlexAlign.Center, alignItems: ItemAlign.Center }) {
          Progress({ value: 10, total: 100, style: this.setProgressStyle })
            .color(this.setProgressColor)
            .width(180)
            .value(this.setValue)
        }
        .width('100%')
        .height('100%')
      }
      .width('100%')
      .constraintSize({ minHeight: 218, maxHeight: 402 })
      .padding({ left: 12, right: 12, top: 22, bottom: 22 })

      Scroll() {
        Column() {
          Flex({ justifyContent: FlexAlign.SpaceBetween, alignItems: ItemAlign.Center }) {
            Text('setProgressColor')
              .fontColor('#182431')
              .fontSize('16fp')
              .fontWeight(FontWeight.Medium)
              .opacity(0.5)
            Column() {
              GetColor({ colorVal: $setProgressColor })
            }.width(48)
          }
          .width('100%')
          .padding({ left: '3%', right: '3%', top: 12, bottom: 12 })
          .borderRadius(24)
          .backgroundColor('#FFFFFF')
          .margin({ top: 8, bottom: 8 })

          Flex({ justifyContent: FlexAlign.SpaceBetween, alignItems: ItemAlign.Center }) {
            Text('progress style')
              .fontColor('#182431')
              .fontSize('16fp')
              .fontWeight(FontWeight.Medium)
              .opacity(0.5)
            Column() {
              Text(`${this.progressStyleString}`)
                .fontColor('#182431')
                .fontSize('16fp')
                .fontWeight(FontWeight.Medium)
            }
            .bindMenu([
              {
                value: 'Linear',
                action: () => {
                  console.info('handle Menu1 select')
                  this.setProgressStyle = ProgressStyle.Linear
                  this.progressStyleString = 'Linear'
                }
              },
              {
                value: 'Eclipse',
                action: () => {
                  console.info('handle Menu2 select')
                  this.setProgressStyle = ProgressStyle.Eclipse
                  this.progressStyleString = 'Eclipse'
                }
              },
              {
                value: 'Ring',
                action: () => {
                  this.setProgressStyle = ProgressStyle.Ring
                  this.progressStyleString = 'Ring'
                }
              },
              {
                value: 'ScaleRing',
                action: () => {
                  this.setProgressStyle = ProgressStyle.ScaleRing
                  this.progressStyleString = 'ScaleRing'
                }
              },
              {
                value: 'Capsule',
                action: () => {
                  this.setProgressStyle = ProgressStyle.Capsule
                  this.progressStyleString = 'Capsule'
                }
              }
            ])
          }
          .width('100%')
          .padding({ left: '3%', right: '3%', top: 12, bottom: 12 })
          .borderRadius(24)
          .backgroundColor('#FFFFFF')
          .margin({ top: 8, bottom: 8 })

          Column() {