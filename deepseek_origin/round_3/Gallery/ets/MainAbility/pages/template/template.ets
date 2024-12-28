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
import { NavigationBar } from "../../common/components/navigationBar"
import { GetColor } from "../../common/components/getColor"

@Entry
@Component
struct BasicComponentTemplate {
  @State value: string = "hello world"
  @State fontColor: string = '#182431';
  @State backgroundColor: string = '#FFB6C1'
  @State setValue: number = 10;
  @State setSliderStyle: SliderStyle = SliderStyle.OutSet;

  build() {
    Flex({ direction: FlexDirection.Column, alignItems: ItemAlign.Center, justifyContent: FlexAlign.Start }) {
      NavigationBar({ title: '样式模板' })
      Scroll() {
        Flex({ direction: FlexDirection.Column, justifyContent: FlexAlign.Center, alignItems: ItemAlign.Center }) {
          Text('显示效果')
            .fontSize(24)
            .fontColor(this.fontColor)
          Text('显示效果')
            .fontSize(24)
            .fontColor(this.fontColor)
          Text('显示效果')
            .fontSize(24)
            .fontColor(this.fontColor)
          Text('显示效果')
            .fontSize(24)
            .fontColor(this.fontColor)
          Text('显示效果')
            .fontSize(24)
            .fontColor(this.fontColor)
          Text('显示效果')
            .fontSize(24)
            .fontColor(this.fontColor)
          Text('显示效果')
            .fontSize(24)
            .fontColor(this.fontColor)
          Text('显示效果')
            .fontSize(24)
            .fontColor(this.fontColor)
          Text('显示效果')
            .fontSize(24)
            .fontColor(this.fontColor)
          Text('显示效果')
            .fontSize(24)
            .fontColor(this.fontColor)
          Text('显示效果')
            .fontSize(24)
            .fontColor(this.fontColor)
          Text('显示效果')
            .fontSize(24)
            .fontColor(this.fontColor)
          Text('显示效果')
            .fontSize(24)
            .fontColor(this.fontColor)
        }
        .width('97%')
        .margin({ right: '3%' })
      }
      .constraintSize({ minHeight: 218, maxHeight: 402 })
      .padding({ left: 12, right: 12, top: 22, bottom: 22 })
      .backgroundColor(this.backgroundColor)

      Scroll() {
        Column() {

          Flex({ justifyContent: FlexAlign.SpaceBetween, alignItems: ItemAlign.Center }) {
            Text('value')
              .fontSize('16fp')
              .fontColor('#182431')
              .opacity(0.5)
              .fontWeight(FontWeight.Medium)

            Text(`${this.value}`)
              .fontSize('16fp')
              .fontColor('#182431')
              .fontWeight(FontWeight.Medium)
              .width('50%')
              .textAlign(TextAlign.End)
              .bindMenu([
                {
                  value: 'hello world',
                  action: () => {
                    this.value = "hello world"
                  }
                },
                {
                  value: 'just do it',
                  action: () => {
                    this.value = "just do it"
                  }
                },
                {
                  value: 'to be No.1',
                  action: () => {
                    this.value = "to be No.1"
                  }
                },
              ])
            

          }
          .padding('3%')
          .borderRadius(24)
          .backgroundColor('#FFFFFF')

          Flex({ justifyContent: FlexAlign.SpaceBetween, alignItems: ItemAlign.Center }) {
            Text('布尔值')
              .fontSize('16fp')
              .fontColor('#182431')
              .opacity(0.5)
              .fontWeight(FontWeight.Medium)
            Toggle({ type: ToggleType.Switch, isOn: false })
              .selectedColor('#007DFF')
              .switchPointColor(0xe5ffffff)
              .onChange((isOn: boolean) => {
                console.info('Component status:' + isOn)
              })
          }
          .padding('3%')
          .borderRadius(24)
          .backgroundColor('#FFFFFF')
          .margin({ top: 8 })

          Column() {