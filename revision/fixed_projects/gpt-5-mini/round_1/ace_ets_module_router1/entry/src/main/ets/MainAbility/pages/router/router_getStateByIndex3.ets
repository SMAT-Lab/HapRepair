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
struct router_getStateByIndex3 {
  @State str: string = ''

  build() {
    Column({ space: 5 }) {

      Text(this.str).id('router_getStateByIndex3_text')

      Button('getStateByIndex')
        .id('router_getStateByIndex3_btn')
        .onClick(() => {
          let options: router.RouterState | undefined = router.getStateByIndex(2)
          if (options) {
            let index = 'index=' + options.index
            let name = 'name=' + options.name
            let path = 'path=' + options.path
            let params = 'params=' + JSON.stringify(options.params)
            this.str = index + ';' + name + ';' + path + ';' + params + ';'
          }
        })

    }.width('100%').height('100%')

  }
}