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
import router from '@ohos.router'
import { BusinessError } from '@ohos.base'
import('./nameDetail9');

class routerParams24 {
  data1:string

  constructor(str:string) {
    this.data1 = str
  }
}

@Entry
@Component
struct replaceNamedRoute10 {
  // str 未发生变化，改为普通变量
  str:string = 'replaceNamedRoute10'
  @State code:string = ''

  build() {
    Column({space:5}) {

      Text(this.str + 'code' + this.code).id('replaceNamedRoute10_text')

      Button('replaceNamedRoute10')
        .id('replaceNamedRoute10_btn')
        .onClick(()=>{
          router.replaceNamedRoute({
            name: 'nameDetail99999999',
            params: new routerParams24('message')
          })
            .then(() => {
            })
            .catch((err:BusinessError) => {
              this.code = err.code + ''
              console.error(`replaceUrl failed, code is ${err.code}, message is ${err.message}`);
            });
        })

    }.width('100%').height('100%')


  }
}