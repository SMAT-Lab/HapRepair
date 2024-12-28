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
struct BorderSample {
  @State borderStyle: string = 'Solid'
  @State borderWidth: number = 18
  @State borderColor: string = 'rgba(0,0,0,1)'
  @State borderRadius: number = 0

  build() {
    Flex({ direction: FlexDirection.Column, alignItems: ItemAlign.Center, justifyContent: FlexAlign.Start }) {
      NavigationBar({ title: '边框设置' })
      Column() {
        Scroll() {
          Flex({ direction: FlexDirection.Column, justifyContent: FlexAlign.Center, alignItems: ItemAlign.Center }) {
            Text(this.borderStyle)
              .textAlign(TextAlign.Center)
              .fontSize(35)
              .width('100%')
              .height('100%')
          }.width(300)
          .height(300)
          .borderRadius(this.borderRadius)
          .borderWidth(this.borderWidth)
          .borderStyle(BorderStyle[this.borderStyle])
          .borderColor(this.borderColor)
        }
      }
      .width('100%')
      .constraintSize({ minHeight: 218, maxHeight: 402 })
      .padding({ left: 12, right: 12, top: 22, bottom: 22 })

      Scroll() {
        Column() {
          Flex({ justifyContent: FlexAlign.SpaceBetween, alignItems: ItemAlign.Center }) {
            Text('borderStyle')
              .fontSize('16fp')
              .opacity(0.5)
              .fontColor('#182431')
              .fontWeight(FontWeight.Medium)
            Column() {
              Text(`${this.borderStyle}`)
                .fontSize('16fp')
                .fontColor('#182431')
                .fontWeight(FontWeight.Medium)
                .width('50%')
                .textAlign(TextAlign.End)
            }
            .bindMenu([
              {
                value: 'Solid',
                action: () => {
                  this.borderStyle = "Solid"
                }
              },
              {
                value: 'Dotted',
                action: () => {
                  this.borderStyle = "Dotted"
                }
              },
              {
                value: 'Dashed',
                action: () => {
                  this.borderStyle = "Dashed"
                }
              },
            ])
          }
          .width('100%')
          .padding({ left: '3%', right: '3%', top: 12, bottom: 12 })
          .borderRadius(24)
          .backgroundColor('#FFFFFF')

          Column() {