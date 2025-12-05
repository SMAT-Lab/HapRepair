/**
 * Copyright (c) 2024 Huawei Device Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the 'License');
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an 'AS IS' BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
@Entry
@Component
struct focusedListItemExample {
  @State arr: number[] = [0, 1, 2, 3, 4, 5, 6]
  @State oneButtonColor: string = '#FFC0CB'

  build() {
    Column() {
      List({ space: 10, initialIndex: 0 }) {
        ForEach(this.arr, (item: number) => {
          ListItem() {
            Text('item-' + item)
              .id('focused_color_' + item)
              .width('100%')
              .height(100)
              .fontSize(16)
              .textAlign(TextAlign.Center)
              .borderRadius(10)
              .backgroundColor(this.oneButtonColor)
              .focusable(true)
              .onFocus(() => {
                this.oneButtonColor = '#FF0000'
              })
              .onBlur(() => {
                this.oneButtonColor = '#FFC0CB'
              })
          }
          
        }, (item: string) => item)
      }.width('90%')
      .height('60%')
      .scrollBar(BarState.Off)

    }.width('100%')
    .height('100%')
    .backgroundColor(0xDCDCDC)
    .padding({ top: 5 })
  }
}