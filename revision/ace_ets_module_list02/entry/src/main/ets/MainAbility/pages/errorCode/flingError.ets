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
import { BusinessError } from '@ohos.base';

@Entry
@Component
struct ListFlingError {
  scroller: Scroller = new Scroller()
  scroller2: Scroller = new Scroller()
  private arr: number[] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
  @State textResult: string = 'Hello world'

  build(){
    Column({ space: 5 }) {
      Text('fling: ' + this.textResult)
        .id('fling_err_text')
        .margin({top:10})
      Button('fling: 401')//401
        .id('fling_401')
        .height('5%')
        .onClick(()=>{
          try {
            this.scroller.fling(undefined)
          } catch (e) {
            this.textResult = (e as BusinessError).code + '';
          }
        })
      Button('fling: 100004') //100004
        .id('fling_100004')
        .height('5%')
        .onClick(()=>{
          try {
            this.scroller2.fling(-1000)
          } catch (error) {
            this.textResult = JSON.stringify(error.code)//'100004'
            console.info('Get error code: %{public}s', JSON.stringify(error.code));
            console.error('error fling failed, error code: ' + error.code + ', error message: ' + error.message)
          }

        })
      List({ initialIndex: 0, scroller: this.scroller }) {
        ForEach(this.arr, (item: number) => {
          ListItem() {
            Text('' + item)
              .width('80%')
              .height('20%')
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
      .height('70%')
      .backgroundColor(0xFAEEE0)
    }
    .width('100%')
    .height('100%')
    .backgroundColor(0xDCDCDC)
    .padding({ top: 5 })
  }
}