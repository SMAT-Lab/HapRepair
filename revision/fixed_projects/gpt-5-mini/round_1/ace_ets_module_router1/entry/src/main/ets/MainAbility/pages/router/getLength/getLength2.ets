
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
struct getLength2 {
  // str 未发生变化，改为普通变量
  str:string = 'getLength2'
  @State sizeValue:string = ''

  build() {
    Column({space:5}) {

      Text(this.str).id('getLength2_text')
      Text('size' + this.sizeValue).id('getLength2_text2')

      Button('getLength')
        .id('getLength2_get')
        .onClick(()=>{
          this.sizeValue = router.getLength()
          AppStorage.setOrCreate('getLength2_size', this.sizeValue)
        })

      Button('pushUrl30')
        .id('getLength2_btn')
        .onClick(async ()=>{
          for(let i=1;i<31;i++){
            try {
             await router.pushUrl({
                url: 'MainAbility/pages/router/getLength/getLength2'
              })
            } catch (err) {
              console.error(`pushUrl failed, code is ${(err as BusinessError).code}, message is ${(err as BusinessError).message}`);
            }
          }
        })

    }.width('100%').height('100%')


  }
}