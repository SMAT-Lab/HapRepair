
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
struct ContainerComponentTemplate {
  @State content: string = "hello world"
  @State backgroundColor: string = '#b6dfff'
  @State value: number = 0
  @State setValue: number = 10;
  @State fontColor: string = '#182431';

  build() {
    Flex({ direction: FlexDirection.Column, alignItems: ItemAlign.Center, justifyContent: FlexAlign.Start }) {
      NavigationBar({ title: '容器组件模板' })

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
      .width('100%')
      .height('45%')
      .backgroundColor(this.backgroundColor)

      Scroll() {
        Column() {

          Flex({ justifyContent: FlexAlign.SpaceBetween, alignItems: ItemAlign.Center }) {
            Text('content选项')
              .fontSize('16fp')
              .fontColor('#182431')
              .opacity(0.5)
              .fontWeight(FontWeight.Medium)

            Text(`${this.content}`)
              .bindMenu([
                {
                  value: 'hello world',
                  action: () => {
                    this.content = "hello world"
                  }
                },
                {
                  value: 'just do it',
                  action: () => {
                    this.content = "just do it"
                  }
                },
                {
                  value: 'to be No.1',
                  action: () => {
                    this.content = "to be No.1"
                  }
                },
              ])
              .fontSize('16fp')
              .fontColor('#182431')
              .fontWeight(FontWeight.Medium)
              .width('50%')
              .textAlign(TextAlign.End)
            

          }
          .padding('3%')
          .borderRadius(24)
          .backgroundColor('#FFFFFF')

          Flex({ justifyContent: FlexAlign.SpaceBetween, alignItems: ItemAlign.Center }) {
            Text('true或者false样式')
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

          Flex({ justifyContent: FlexAlign.SpaceBetween, alignItems: ItemAlign.Center }) {
            Text('数字选项样式')
              .fontSize('16fp')
              .fontColor('#182431')
              .opacity(0.5)
              .fontWeight(FontWeight.Medium)
            Column() {
              Counter() {
                Text(this.value.toString())
              }
              .onInc(() => {
                this.value++
              })
              .onDec(() => {
                this.value--
              })
            }
          }
          .padding('3%')
          .borderRadius(24)
          .backgroundColor('#FFFFFF')
          .margin({ top: 8 })

          Flex({ justifyContent: FlexAlign.SpaceBetween, alignItems: ItemAlign.Center }) {
            Button('Ok', { type: ButtonType.Capsule, stateEffect: true }).backgroundColor(0x317aff).width(90)
              .onClick((event: ClickEvent) => {
                console.info('点击按钮1')
              })
            Button('Ok', { type: ButtonType.Capsule, stateEffect: true }).backgroundColor(0x317aff).width(90)
              .onClick((event: ClickEvent) => {
                console.info('点击按钮2')
              })
          }
          .padding('3%')
          .borderRadius(24)
          .backgroundColor('#FFFFFF')
          .margin({ top: 8 })

          Column() {