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
struct comSizeAbilityTest_19 {
  private arrayA: string[] = ['安']
  private arrayB: string[] = ['卜', '白', '包', '毕', '丙']
  private arrayC: string[] = ['曹', '成', '陈', '催']
  private value: string[] = ['#', 'A', 'B', 'C']

  build() {
    Column() {
      Stack({ alignContent: Alignment.Start }) {
        Row() {
          List({}) {
            ForEach(this.arrayA, (item: string) => {
              ListItem() {
                Text(item)
                  .width('50%')
                  .height('5%')
                  .fontSize(30)
                  .textAlign(TextAlign.Center)
              }.editable(true)
            }, (item: string) => item)
            ForEach(this.arrayB, (item: string) => {
              ListItem() {
                Text(item)
                  .width('50%')
                  .height('5%')
                  .fontSize(30)
                  .textAlign(TextAlign.Center)
              }.editable(true)
            }, (item: string) => item)
            ForEach(this.arrayC, (item: string) => {
              ListItem() {
                Text(item)
                  .width('50%')
                  .height('5%')
                  .fontSize(30)
                  .textAlign(TextAlign.Center)
              }.editable(true)
            }, (item: string) => item)
          }
          .width('30%')
          .backgroundColor('green')

          AlphabetIndexer({ arrayValue: this.value, selected: 0 })
            .size({ width: 80, height: 80 })
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
            .key("AlphabetIndexer_Size")
            .backgroundColor('red')
            .selectedColor(0xFFFFFF)// 选中项文本颜色
            .popupColor(0xFFFAF0)// 弹出框文本颜色
            .selectedBackgroundColor(0xCCCCCC)// 选中项背景颜色
            .popupBackground(0xD2B48C)// 弹出框背景颜色
            .usingPopup(true)// 是否显示弹出框
            .selectedFont({ size: 16, weight: FontWeight.Bolder })// 选中项字体样式
            .popupFont({ size: 30, weight: FontWeight.Bolder })// 弹出框内容的字体样式
            .itemSize(28)// 每一项的尺寸大小
            .alignStyle(IndexerAlign.Left)// 弹出框在索引条右侧弹出
            .onSelect((index: number) => {
              console.info(this.value[index] + ' Selected!')
            })
            .onRequestPopupData((index: number) => {
              if (this.value[index] == 'A') {
                return this.arrayA // 当选中A时，弹出框里面的提示文本列表显示A对应的列表arrayA，选中B、C、L时也同样
              } else if (this.value[index] == 'B') {
                return this.arrayB
              } else if (this.value[index] == 'C') {
                return this.arrayC
              } else {
                return [] // 选中其余子母项时，提示文本列表为空
              }
            })
            .onPopupSelect((index: number) => {
              console.info('onPopupSelected:' + index)
            })
        }
        .width('100%')
      }
    }.key('AlphabetIndex_Father')
  }
}