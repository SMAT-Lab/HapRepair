
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

class MyDataSource1 implements IDataSource {
  private list: number[] = []

  constructor(list: number[]) {
    this.list = list
  }

  totalCount(): number {
    return this.list.length
  }

  getData(index: number): number {
    return this.list[index]
  }

  registerDataChangeListener(listener: DataChangeListener): void {
  }

  unregisterDataChangeListener() {
  }
}

@Entry
@Component
struct comSizeAbilityTest_28 {
  private swiperController: SwiperController = new SwiperController()
  private data: MyDataSource1 = new MyDataSource1([])

  aboutToAppear(): void {
    let list: number[] = []
    for (let i = 1; i <= 10; i++) {
      list.push(i);
    }
    this.data = new MyDataSource1(list)
@Reusable
@Component
struct SwiperItemComponent {
  item: string = ''
  build() {
    Text(this.item.toString())
      .width('90%')
      .height(160)
      .backgroundColor(0xAFEEEE)
      .textAlign(TextAlign.Center)
      .fontSize(30)
  }
}
  }

  build() {
    Scroll() {
      Column({ space: 5 }) {
        Swiper(this.swiperController) {
          LazyForEach(this.data, (item: string) => {
            SwiperItemComponent({
              item: item
            })
          }, (item: string) => item)
        }
        .size({ width: 250, height: 100 })
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
        .key("Swiper_Size")
        .cachedCount(2)
        .index(1)
        .autoPlay(true)
        .interval(4000)
        .indicator(true)
        .loop(true)
        .duration(1000)
        .itemSpace(0)
        .displayArrow({
          showBackground: true,
          isSidebarMiddle: true,
          backgroundSize: 24,
          backgroundColor: Color.White,
          arrowSize: 18,
          arrowColor: Color.Blue
        }, false)
        .curve(Curve.Linear)
      }.width('100%').margin({ top: 5 }).key('Swiper_Father')
    }.height('100%')
  }
}

