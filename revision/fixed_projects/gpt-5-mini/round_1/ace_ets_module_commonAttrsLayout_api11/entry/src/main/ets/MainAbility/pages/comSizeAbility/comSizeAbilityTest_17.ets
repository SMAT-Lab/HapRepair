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

class bottom {
  bottom: number = 50
}

let bott: bottom = new bottom()

@Entry
@Component
struct comSizeAbilityTest_17 {
  private select: number = 1
  private apfruits: string[] = ['选项1', '选项2', '选项3', '选项4']
  scroller: Scroller = new Scroller()

  build() {
    Column() {
      Row() {
        Button('to_1').onClick(()=>{
          this.scroller.scrollToIndex(0)
        }).key('comSizeAbilityTest_17_to_1')

        Button('to_2').onClick(()=>{
          this.scroller.scrollToIndex(1)
        }).key('comSizeAbilityTest_17_to_2')

        Button('to_3').onClick(()=>{
          this.scroller.scrollToIndex(2)
        }).key('comSizeAbilityTest_17_to_3')
      }

      List({ space: 5, initialIndex: 0, scroller: this.scroller }) {
        ListItem() {
          TextPicker({ range: this.apfruits, selected: this.select })
            .onChange((value: string | string[], index: number | number[]) => {
              console.info('Picker item changed, value: ' + value + ', index: ' + index)
            })
            .margin(bott)
            .backgroundColor(0XBDDB69)
            .key("TextPicker_Size")
            .size({ width: 130, height: 110 })
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
        }.key('TextPicker_Father')

        ListItem() {
          Text('显示文本组件')
            .key("Text_Size")
            .textAlign(TextAlign.Center)
            .backgroundColor(0XBDDB69)
            .size({ width: 155, height: 100 })
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
        }.key('Text_Father')

        ListItem() {
          TextInput({ text: '文本内容' })
            .backgroundColor(0XBDDB69)
            .key("TextInput_Size")
            .size({ width: 135, height: 110 })
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
        }.key('TextInput_Father')
      }.width('100%')
    }.height('100%')
  }
}


