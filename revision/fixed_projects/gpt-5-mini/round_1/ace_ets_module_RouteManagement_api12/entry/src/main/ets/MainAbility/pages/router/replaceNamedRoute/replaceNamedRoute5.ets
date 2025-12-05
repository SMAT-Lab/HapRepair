
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
import('./nameDetail5');

class routerParams19 {
  data1:string

  constructor(str:string) {
    this.data1 = str
  }
}

@Entry
@Component
struct replaceNamedRoute5 {
  // str 未发生变化，改为普通变量
  str: string = 'replaceNamedRoute5'
  @State code:string = ''
  build() {
    Column({space:5}) {

      Text(this.str).id('replaceNamedRoute5_text')

      Button('replaceNamedRoute5')
        .id('replaceNamedRoute5_btn')
        .onClick(()=>{
          router.replaceNamedRoute({
            name: 'nameDetail5',
            params: new routerParams19('message')
          })
            .then(() => {
            })
            .catch((err:BusinessError) => {
              this.code = '' + err.code
              console.error(`replaceNamedRoute failed, code is ${err.code}, message is ${err.message}`);
            });
        })

    }.width('100%').height('100%')


  }
}