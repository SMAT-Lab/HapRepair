
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
struct back3 {
  // str 未发生变化，改为普通变量
  str:string = 'back3'

  build() {
    Column({space:5}) {

      Text(this.str).id('back3_text')

      Button('back')
        .id('back3_btn')
        .onClick(()=>{
          router.back({url:'MainAbility/pages/router/back/abccccc'})
        })


    }.width('100%').height('100%')


  }
}