
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

class routerParams22 {
  data1:string

  constructor(str:string) {
    this.data1 = str
  }
}

@Entry({ routeName: 'replaceNamedRoute8' })
@Component
struct replaceNamedRoute8 {
  // 直接使用一般变量即可
  str:string = 'replaceNamedRoute8'
  @State code:string = ''
  @State text:string = '123'


  aboutToAppear(): void {
    AppStorage.setOrCreate('replaceNamedRoute8_mes', AppStorage.get('replaceNamedRoute8_mes') + ';replaceNamedRoute8_aboutToAppear')
  }

  build() {
    Column({space:5}) {

      Text(this.str).id('replaceNamedRoute8_text')

      Text(this.text).id('replaceNamedRoute8_text2')

      Button('getText')
        .id('replaceNamedRoute8_get')
        .fontSize(14)
        .onClick(() => {
          this.text = AppStorage.get('replaceNamedRoute8_mes') + ''
        })

      Button('replaceNamedRoute8')
        .id('replaceNamedRoute8_btn')
        .onClick(()=>{
          router.replaceNamedRoute({
            name: 'replaceNamedRoute8',
            params: new routerParams22('message')
          }, router.RouterMode.Standard)
            .then(() => {
              console.log('replaceNamedRoute success');
            })
            .catch((err:BusinessError) => {
              this.code = '' + err.code
              console.error(`replaceNamedRoute failed, code is ${err.code}, message is ${err.message}`);
            });
        })

    }.width('100%').height('100%')


  }
}