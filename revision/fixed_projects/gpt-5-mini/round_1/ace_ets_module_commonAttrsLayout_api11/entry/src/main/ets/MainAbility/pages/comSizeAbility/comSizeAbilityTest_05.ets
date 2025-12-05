
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
struct comSizeAbilityTest_05 {
  @State bgColors: Color[] =
    [Color.Red, Color.Orange, Color.Yellow, Color.Green, Color.Pink, Color.Grey, Color.Blue, Color.Brown, Color.Black]
  // currentBp 未发生变化，改为普通变量
  currentBp: string = 'unknown'
  @State Number: String[] = ['0', '1', '2', '3', '4']
  scroller: Scroller = new Scroller()

  build() {
    Column() {
      Row() {
        Button('to_1').onClick(()=>{
          this.scroller.scrollToIndex(0)
        }).key('comSizeAbilityTest_05_to_1')

        Button('to_2').onClick(()=>{
          this.scroller.scrollToIndex(1)
        }).key('comSizeAbilityTest_05_to_2')

        Button('to_3').onClick(()=>{
          this.scroller.scrollToIndex(2)
        }).key('comSizeAbilityTest_05_to_3')
      }

      List({ space: 5, initialIndex: 0, scroller: this.scroller }) {
        ListItem() {
          Gauge({ value: 60 })
            .colors([[0x317AF7, 1], [0x5BA854, 1], [0xE08C3A, 1], [0x9C554B, 1]])
            .key("Gauge_Size")
            .size({ width: 140, height: 140 })

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
        }

        ListItem() {
          GridRow({
            columns: 3,
            gutter: { x: 5, y: 5 },
            breakpoints: {
              value: ["400vp", "400vp", "400vp"],
              reference: BreakpointsReference.WindowSize
            },
            direction: GridRowDirection.Row
          }) {
            ForEach(this.bgColors, (color: Color) => {
              GridCol({
                span: {
                  xs: 1,
                  sm: 2,
                  md: 3,
                  lg: 4
                },
                offset: 0,
                order: 0
              }) {
                Row().width("100%").height("20vp")
              }.borderColor(color).borderWidth(2)
            })
          }
          .backgroundColor('green')
          .key("GridRow_Size")
          .size({ width: 140, height: 140 })
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
        }

        ListItem() {
          Grid() {
            ForEach(this.Number, (day: string) => {
              ForEach(this.Number, (day: string) => {
                GridItem() {
                  Text(day)
                    .fontSize(16)
                    .backgroundColor(0xF9CF93)
                    .width('100%')
                    .height('100%')
                    .textAlign(TextAlign.Center)
                }
              }, (day: string) => day)
            }, (day: string) => day)
          }
          .columnsTemplate('1fr 1fr 1fr 1fr 1fr')
          .rowsTemplate('1fr 1fr 1fr 1fr 1fr')
          .columnsGap(10)
          .rowsGap(10)
          .key("Grid_Size")
          .size({ width: 360, height: 240 })
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
        }
      }.width('100%')
    }.height('100%')
  }
}