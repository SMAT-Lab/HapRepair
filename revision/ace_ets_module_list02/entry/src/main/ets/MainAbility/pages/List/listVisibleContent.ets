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
import promptAction from '@ohos.promptAction'
class BasicDataSource <T> implements IDataSource {
  private listeners: DataChangeListener[] = [];
  private originDataArray: string[] = [];

  public totalCount(): number {
    return 0;
  }

  public getData(index: number): string {
    return this.originDataArray[index];
  }

  registerDataChangeListener(listener: DataChangeListener): void {
    if (this.listeners.indexOf(listener) < 0) {
      console.info('add listener');
      this.listeners.push(listener);
    }
  }

  unregisterDataChangeListener(listener: DataChangeListener): void {
    const pos = this.listeners.indexOf(listener);
    if (pos >= 0) {
      console.info('remove listener');
      this.listeners.splice(pos, 1);
    }
  }

  notifyDataReload(): void {
    this.listeners.forEach(listener => {
      listener.onDataReloaded();
    })
  }

  notifyDataAdd(index: number): void {
    this.listeners.forEach(listener => {
      listener.onDataAdd(index);
    })
  }

  notifyDataChange(index: number): void {
    this.listeners.forEach(listener => {
      listener.onDataChange(index);
    })
  }

  notifyDataDelete(index: number): void {
    this.listeners.forEach(listener => {
      listener.onDataDelete(index);
    })
  }

  notifyDataMove(from: number, to: number): void {
    this.listeners.forEach(listener => {
      listener.onDataMove(from, to);
    })
  }
}

class MyDataSource <T> extends BasicDataSource<T> {
  private dataArray: string[] = [];

  public totalCount(): number {
    return this.dataArray.length;
  }

  public getData(index: number): string {
    return this.dataArray[index];
  }

  public addData(index: number, data: string): void {
    this.dataArray.splice(index, 0, data);
    this.notifyDataAdd(this.totalCount() - 1)
  }

  public dataReload(): void {
    this.notifyDataReload()
  }

  public pushData(data: string): void {
    this.dataArray.push(data);
    this.notifyDataAdd(this.dataArray.length - 1);
  }

  public deleteData(index: number): void {
    this.dataArray.splice(index, 1);
    this.notifyDataAdd(this.totalCount())
  }
}

@Entry
@Component
struct listVisibleContent {
  private dataBean: MyDataSource<string> = new MyDataSource<string>();
  scroller: Scroller = new Scroller()
  private firstViewIndex: number = 0
  private lastViewIndex: number = 0
  @State leg: number = 0

  aboutToAppear() {
    for (let i = 1; i <= 50; i++) {
      this.dataBean.pushData(i + '')
    }
  }

  build() {
    Column(){
      Button('scrollTo')
        .id('scrollTo_y')
        .onClick(() => {
          this.scroller.scrollTo({ xOffset: 0, yOffset: vp2px(200) })
        })
      Button('deleteItem')
        .id('deleteItem_index')
        .onClick(() => {
          if (this.firstViewIndex == 0) {
            promptAction.showToast({
              message: '容器上方没有位置删除数据',
              duration: 3000
            })
            return
          }
          this.dataBean.deleteData(this.firstViewIndex - 1)
          this.leg = this.dataBean.totalCount()
        })
      Text('length: ' + this.leg)
        .id('list_num')

      List({ scroller: this.scroller }) {
        LazyForEach(this.dataBean, (item: string, itemIndex: number) => {
          ListItem({ style: ListItemStyle.CARD }) {
            Text('Item:'+ item + ', itemIndex:' + itemIndex)
          }
          .onAppear(() => {
            console.info('onAppear_item:' + item + ', itemIndex:' + itemIndex)
          })
          .onDisAppear(() => {
            console.info('onDisAppear_item:' + item + ', itemIndex:' + itemIndex)
          })
          .height(65)
          .margin(10)
          .borderWidth(2)
        }, (item: string, index: number) => item + index)
      }
      .id('list1_visible')
      .width('100%')
      .height('60%')
      .backgroundColor(0xDCDCDC)
      .maintainVisibleContentPosition(false)
      .borderWidth(1)
      .divider({ strokeWidth: 1, color: Color.Blue })
      .onScrollIndex((firstIndex: number, lastIndex: number, centerIndex: number) => {
        console.info('first' + firstIndex)
        console.info('last' + lastIndex)
        console.info('center' + centerIndex)
        this.firstViewIndex = firstIndex
        this.lastViewIndex = lastIndex
      })
      .onScrollVisibleContentChange((start: VisibleListContentInfo, end: VisibleListContentInfo) => {
        console.log(' start index: ' + start.index +
          ' start item group area: ' + start.itemGroupArea +
          ' start index in group: ' + start.itemIndexInGroup)
        console.log(' end index: ' + end.index +
          ' end item group area: ' + end.itemGroupArea +
          ' end index in group: ' + end.itemIndexInGroup)
      })
      .onDidScroll((scrollOffset: number, scrollState: ScrollState) => {
        console.info(`onScroll scrollState = ScrollState` + scrollState + `, scrollOffset = ` + scrollOffset)
      })

    }
    .width('100%')
    .height('100%')

  }
}