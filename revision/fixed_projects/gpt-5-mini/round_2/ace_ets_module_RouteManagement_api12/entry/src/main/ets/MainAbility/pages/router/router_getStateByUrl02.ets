
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
struct router_getStateByUrl02 {
  @State str: string = ''

  build() {
    Column({ space: 5 }) {

      Text(this.str).id('router_getStateByUrl02_text')

      Button('getStateByUrl')
        .id('router_getStateByUrl02_btn')
        .onClick(() => {
          let options: Array<router.RouterState> = router.getStateByUrl('MainAbility/pages/router/router_getStateByUrl02')
          let str = this.str;
          for (let i: number = 0; i < options.length; i++) {
            let index = 'index=' + options[i].index
            let name = 'name=' + options[i].name
            let path = 'path=' + options[i].path
            let params = 'params=' + JSON.stringify(options[i].params)
            str = str + index + ';' + name + ';' + path + ';' + params + ';'
          }
          this.str = str;
        })

    }.width('100%').height('100%')

  }
}