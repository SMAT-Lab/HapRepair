

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

@Entry
@Component
struct getState01 {
  // str 未发生变化，改为普通变量
  str: string = 'getState01'
  @State text:string = ''

  build() {
    Column({space:5}) {

      Text(this.str).id('getState01_text')
      Text(this.text).id('getState01_text2')

      Button('getState')
        .id('getState01_btn')
        .onClick(()=>{
          let options:router.RouterState | undefined = router.getState()
          if(options){
            let index = 'index=' + options.index
            let name = 'name=' + options.name
            let path = 'path=' + options.path
            let params = 'params=' + JSON.stringify(options.params)
            this.text = index + ';'+ name +';'+ path + ';'+params + ';'
          }
        })


    }.width('100%').height('100%')


  }
}