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
struct router_back12 {
  // str 未发生变化，改为普通变量
  str: string = 'back12'

  build() {
    Column({ space: 5 }) {

      Text(this.str).id('router_back12_text')

      Button('go to second')
        .id('router_back12_btn2')
        .onClick(() => {
          router.pushUrl({ url: "MainAbility/pages/router/router_back012" })
        })
    }.width('100%').height('100%')
  }
}