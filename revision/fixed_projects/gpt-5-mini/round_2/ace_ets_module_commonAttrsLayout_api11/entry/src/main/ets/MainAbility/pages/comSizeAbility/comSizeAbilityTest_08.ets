
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
struct comSizeAbilityTest_08 {
  private NavRouterArr: number[] = [0]
  isActive: boolean = false
  @State dex: number = 0

  build() {
    Scroll() {
      Column({ space: 20 }) {
        Column() {
          Navigation() {
            List({ space: 0, initialIndex: 0 }) {
              ForEach(this.NavRouterArr, (item: number, index: number = 0) => {
                ListItem() {
                  NavRouter() {
                    Row() {
                      Text(`NavRouter${item + 1}`)
                        .fontSize(22)
                        .fontWeight(500)
                        .textAlign(TextAlign.Center)
                    }
                    .height(50)

                    NavDestination() {
                      Text(`我是第${item + 1}页内容`).fontSize(50)
                      Row() {
                        Text('NavRouter跳转页').fontSize(20)
                      }.padding({ left: 15 })
                    }.backgroundColor('#ccc').height(200).width(300)
                  }
                  .key("Navigation_Size")
                  .onStateChange((isActivated: boolean) => {
                    this.dex = index
                  })
                }
              }, (item: number) => item.toString())
            }
          }
          .backgroundColor('red')
          .key("Navigation_Size")
          .size({ width: 100, height: 100 })
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

        Column() {
          Navigation() {
            ForEach(this.NavRouterArr, (item: number, index: number = 0) => {
              NavRouter() {
                Row() {
                  Text(`NavRouter${item + 1}`)
                    .fontSize(22)
                    .textAlign(TextAlign.Center)
                }
                .height(50)
                .backgroundColor(this.dex === index ? '#ccc' : '#fff')
              }
              .backgroundColor('red')
              .key("NavRouter_Size")
              .size({ width: 100, height: 100 })
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
            }, (item: number) => item.toString())
          }
        }
        .backgroundColor('blue')
        .height('30%')
      }.width('100%')
    }.height('100%')
  }
}