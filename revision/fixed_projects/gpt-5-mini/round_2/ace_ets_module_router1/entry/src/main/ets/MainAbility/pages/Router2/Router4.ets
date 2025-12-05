
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
// 在second页面中接收传递过来的参数
import router from '@ohos.router'


class routerParams {
  data1:string
  data2:number[]

  constructor(data1:string, data2:number[]) {
    this.data1 = data1
    this.data2 = data2
  }
}

@Entry
@Component
struct Second4 {
  private content: string = "这是第二页"
  // 直接使用一般变量即可
  data1: string = (router.getParams() as routerParams).data1
  @State data2: number[] = (router.getParams() as routerParams).data2

  build() {
    Column() {
      Text('' + this.content)
      Text('' + this.data1).id('arrayData1')
      Text('' + this.data2[0]).id('arrayData2')
    }.width('100%')


  }
}