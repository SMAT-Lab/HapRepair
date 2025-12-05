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
@Entry
@Component
struct ListExample_get {
  private arr: number[] = []
  private scroller: ListScroller = new ListScroller()
  @State listSpace: number = 10
  @State listChildrenSize: ChildrenMainSize = new ChildrenMainSize(100)
  aboutToAppear(){
    // 初始化数据源。
    for (let i = 0; i < 10; i++) {
      this.arr.push(i)
    }
    // 前5个item的主轴大小不是默认大小100，因此需要通过ChildrenMainSize通知List。
    this.listChildrenSize.splice(0, 5, [300, 300, 300, 300, 300])
  }
  @State get: string = 'get'
  build() {
    Column() {
      Text(this.get).id('get')
      List({ space: this.listSpace, initialIndex: 4, scroller: this.scroller }) {
        ForEach(this.arr, (item: number) => {
          ListItem() {
            Text('item-' + item)
              .height( item < 5 ? 300 : this.listChildrenSize.childDefaultSize)
              .width('90%')
              .fontSize(16)
              .textAlign(TextAlign.Center)
              .borderRadius(10)
              .backgroundColor(0xFFFFFF)
          }
        }, (item: string) => item)
      }
      .backgroundColor(Color.Gray)
      .layoutWeight(1)
      .scrollBar(BarState.On)
      .childrenMainSize(this.listChildrenSize)
      .alignListItem(ListItemAlign.Center)
      Row({ space: 10 }){
        Button() { Text('item size - 100') }.onClick(()=>{
          this.listChildrenSize.childDefaultSize -= 100
          if (this.listChildrenSize.childDefaultSize === 0) {
            this.get = 'get childDefaultSize:0'
          }
        }).height('50%').width('30%').id('get_childDefaultSize')
      }.height('20%')
    }
  }
}