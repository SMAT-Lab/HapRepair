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
struct router_back15a {
  @State index: string = 's'

  build() {
    Column({ space: 5 }) {

      Text('back15a')
      Text(this.index).id('router_back15a_text')

      Button('go to next page')
        .id('router_back15a_btn1')
        .onClick(() => {
          router.pushUrl({ url: "MainAbility/pages/router/router_back15b" })
        })


      Button('get page index')
        .id('router_back15a_btn2')
        .onClick(() => {
          let page = router.getState();
          this.index = String(page.index);
        })

    }.width('100%').height('100%')

  }
}