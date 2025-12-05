
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
struct comSizeAbilityTest_13 {
  // data 未发生变化，改为普通变量
  data: string = '<h1 style="text-align: center;">h1标题</h1>'
  ScrollSize: number[] = [261, 140]
  scroller: Scroller = new Scroller()
  private arr: number[] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  space: number = 8
  scroller1: Scroller = new Scroller()

  build() {
    Column() {
      Row() {
        Button('to_1').onClick(()=>{
          this.scroller1.scrollToIndex(0)
        }).key('comSizeAbilityTest_13_to_1')

        Button('to_2').onClick(()=>{
          this.scroller1.scrollToIndex(1)
        }).key('comSizeAbilityTest_13_to_2')

        Button('to_3').onClick(()=>{
          this.scroller1.scrollToIndex(2)
        }).key('comSizeAbilityTest_13_to_3')
      }

      List({ space: 5, initialIndex: 0, scroller: this.scroller1 }) {
        ListItem() {
          RichText(this.data)
            .backgroundColor(0XBDDB69)
            .key("RichText_Size")
            .size({ width: 210, height: 132 })
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
        }.key('RichText_Father')

        ListItem() {
          Scroll(this.scroller) {
            Column() {
              ForEach(this.arr, (item: number) => {
                Text(item.toString())
                  .width('90%')
                  .height(100)
                  .backgroundColor(0xFFFFFF)
                  .borderRadius(15)
                  .textAlign(TextAlign.Center)
                  .margin({ top: 10 })
              }, (item: string) => item)
            }.width('100%')
          }
          .backgroundColor(0XBDDB69)
          .scrollable(ScrollDirection.Vertical)
          .scrollBar(BarState.On)
          .scrollBarColor(Color.Gray)
          .scrollBarWidth(10)
          .key("Scroll_Size")
          .size({ width: '70%', height: '70%' })
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
        }.key('Scroll_Father').width('100%').height('40%')

        ListItem() {
          Search()
            .backgroundColor(0XBDDB69)
            .key("Search_Size")
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
        }.key('Search_Father')
      }.width('100%')
    }.height('100%')
  }
}