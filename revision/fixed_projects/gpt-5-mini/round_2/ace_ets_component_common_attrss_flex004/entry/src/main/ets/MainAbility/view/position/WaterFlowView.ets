
/*
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

@Component
export struct WaterFlowView {
  @Link _position: Position;
  private componentKey: string;
  minSize: number = 50
  maxSize: number = 100
  fontSize: number = 24
  @State colors: number[] = [0xFFC0CB, 0xDA70D6, 0x6B8E23, 0x6A5ACD, 0x00FFFF, 0x00FF7F]
  scroller: Scroller = new Scroller()
  private itemWidthArray: number[] = []
  private itemHeightArray: number[] = []

  getSize() {
    let ret = Math.floor(Math.random() * this.maxSize)
    return (ret > this.minSize ? ret : this.minSize)
  }

  getItemSizeArray() {
    for (let i = 0; i < 100; i++) {
      this.itemWidthArray.push(this.getSize())
      this.itemHeightArray.push(this.getSize())
    }
  }

  aboutToAppear() {
    this.getItemSizeArray()
  }

  @Builder
  itemFoot() {
    Text(`Footer`)
      .fontSize(10)
      .backgroundColor(Color.Red)
      .width(50)
      .height(50)
      .align(Alignment.Center)
      .margin({ top: 2 })
  }

  build() {
    WaterFlow({ footer: this.itemFoot.bind(this), scroller: this.scroller }) {
      ForEach([1, 2, 3, 4, 5, 6, 7, 8, 9], (item: number) => {
        FlowItem() {
          Text('N' + item).fontSize(12).height('16')
        }
        .width(this.itemWidthArray[item])
        .height(this.itemHeightArray[item])
        .backgroundColor(this.colors[item % 5])
      }, (item: number) => item.toString())
    }
    .columnsTemplate('1fr 1fr 1fr 1fr')
    .itemConstraintSize({
      minWidth: 0,
      maxWidth: '100%',
      minHeight: 0,
      maxHeight: '100%'
    })
    .columnsGap(10)
    .rowsGap(5)
    .onReachStart(() => {
      console.info('onReachStart')
    })
    .onReachEnd(() => {
      console.info('onReachEnd')
    })
    .backgroundColor(0xFAEEE0)
    .width('100%')
    .height('80%')
    .layoutDirection(FlexDirection.Column)
    .position(this._position)
    .key(this.componentKey)
  }
}