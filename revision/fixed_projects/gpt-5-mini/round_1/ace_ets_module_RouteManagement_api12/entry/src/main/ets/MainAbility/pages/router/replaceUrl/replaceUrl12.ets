
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

class routerParams8 {
  data1:string

  constructor(str:string) {
    this.data1 = str
  }
}

@Entry
@Component
struct replaceUrl12 {
  // str 未发生变化，改为普通变量
  str:string = 'replaceUrl12'
  @State code:string = ''
  @State text:string = '123'


  aboutToAppear(): void {
    AppStorage.setOrCreate('replaceUrl12_mes', AppStorage.get('replaceUrl12_mes') + ';replaceUrl12_aboutToAppear')
  }

  build() {
    Column({space:5}) {

      Text(this.str).id('replaceUrl12_text')

      Text(this.text).id('replaceUrl12_text2')

      Button('getText')
        .id('replaceUrl12_get')
        .fontSize(14)
        .onClick(() => {
          this.text = AppStorage.get('replaceUrl12_mes') + ''
        })

      Button('replaceUrl12')
        .id('replaceUrl12_btn')
        .onClick(()=>{
          router.replaceUrl({
            url: 'MainAbility/pages/router/replaceUrl/replaceUrl12',
            params: new routerParams8('message')
          }, router.RouterMode.Single)
            .then(() => {
              console.log('replaceUrl success');
            })
            .catch((err:BusinessError) => {
              this.code = '' + err.code
              console.error(`replaceUrl failed, code is ${err.code}, message is ${err.message}`);
            });
        })

    }.width('100%').height('100%')


  }
}