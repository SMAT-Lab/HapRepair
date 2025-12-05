

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

@Entry
@Component
struct back9 {
  // str 未发生变化，改为普通变量
  str: string = 'back9'

  build() {
    Column({space:5}) {

      Text(this.str).id('back9_text')

      Button('go to second')
        .id('back9_btn')
        .onClick(()=>{
          router.pushUrl({url:"MainAbility/pages/router/back/back09"})
        })

    }.width('100%').height('100%')


  }
}