
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

class routerParams6 {
  data1:string

  constructor(str:string) {
    this.data1 = str
  }
}

@Entry
@Component
struct replaceUrl6 {
  // str 未发生变化，改为普通变量
  str: string = 'replaceUrl6'
  @State code:string = ''

  build() {
    Column({space:5}) {

      Text(this.str + 'code' + this.code).id('replaceUrl6_text')

      Button('replaceUrl6')
        .id('replaceUrl6_btn')
        .onClick(()=>{
          router.replaceUrl({
            url: 'MainAbility/pages/router/replaceUrl/detail11111',
            params: new routerParams6('message')
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