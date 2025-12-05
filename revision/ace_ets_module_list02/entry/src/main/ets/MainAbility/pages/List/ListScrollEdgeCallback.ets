/**
 * Copyright (c) 2024 Huawei Device Co., Ltd.
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
import events_emitter from '@ohos.events.emitter'

@Entry
@Component
struct ListScrollEdgeCallback {
  @State numbers: String[] = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
  private scroller: Scroller = new Scroller()
  private arr: number[] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
  @State text1: string = ''
  build(){
    Column({ space: 5 }) {
      Text('Scroller组件绑定至平铺的List容器组件').fontColor(0x000000)
        .fontSize(16).width('90%')
      Text(this.text1).id('ListTextByScrollEdge01')
      Button('点我滚动')
        .id('ListScrollEdgeCallback01')
        .onClick(() => { // 这里
          this.scroller.scrollEdge(Edge.Bottom, { velocity: 0 })
          this.text1 = 'ScrollEdgeSuccess'
        })
        .width('100%')
      List({ space: 20, initialIndex: 0, scroller: this.scroller }) {
        ForEach(this.arr, (item: number) => {
          ListItem() {
            Text('' + item)
              .width('80%')
              .height(100)
              .fontSize(16)
              .fontColor('#ffffff')
              .textAlign(TextAlign.Center)
              .borderRadius(10)
              .backgroundColor('#106836')
          }
        }, (item: string) => item)
      }
      .alignListItem(ListItemAlign.Center)
      .friction(0.6)
      .edgeEffect(EdgeEffect.Fade)
      .scrollBar(BarState.On)
      .width('90%')
      .backgroundColor(0xFAEEE0)
      .height('70%')

    }
    .width('100%')
    .height('100%')
    .backgroundColor(0xDCDCDC)
    .padding({ top: 5 })
  }
}
