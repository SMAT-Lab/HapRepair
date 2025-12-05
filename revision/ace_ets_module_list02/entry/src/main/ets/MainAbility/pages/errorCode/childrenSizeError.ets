/**
 * Copyright (c) 2025 Huawei Device Co., Ltd.
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
struct childrenMainSizeError {
  private arr: number[] = [0, 1, 2, 3, 4, 5, 6, 7]
  private scroller: ListScroller = new ListScroller()
  @State listChildrenSize: ChildrenMainSize = new ChildrenMainSize(20)
  @State errorSplice401: string = 'splice';
  @State errorUpdate401: string = 'update';
  @State errorDef401: string = 'childDefaultSize';

  build() {
    Column() {
      Text('spliceError: ' + this.errorSplice401)//401
        .id('splice_err_401')
        .margin({top:5})

      Button('splice: 401')
        .id('splice_btn_401')
        .height('5%')
        .onClick(()=>{
          try {
            this.listChildrenSize.splice(-1)
          } catch (e) {
            this.errorSplice401 = (e as BusinessError).code + '';
          }
        })
      Text('updateError: ' + this.errorUpdate401)//401
        .id('update_err_401')
        .margin({top:5})

      Button('update: 401')
        .id('update_btn_401')
        .height('5%')
        .onClick(()=>{
          try {
            this.listChildrenSize.update(-1, 0)
          } catch (e) {
            this.errorUpdate401 = (e as BusinessError).code + '';
          }
        })
      Text('childDefaultSize: ' + this.errorDef401)//401
        .id('childDefaultSize_err_401')
        .margin({top:5})

      Button('childDefaultSize: -1')
        .id('childDefaultSize_btn_401')
        .height('5%')
        .onClick(()=>{
          try {
            this.listChildrenSize.childDefaultSize = -1
          } catch (e) {
            this.errorDef401 = (e as BusinessError).code + '';
          }
        })
      List({ space: 2, scroller: this.scroller }) {
        ForEach(this.arr, (item: number) => {
          ListItem() {
            Text('item-' + item)
              .width('80%')
              .height('5%')
              .textAlign(TextAlign.Center)
          }.backgroundColor(Color.Pink)
        })
      }
      .childrenMainSize(this.listChildrenSize)
    }
  }
}