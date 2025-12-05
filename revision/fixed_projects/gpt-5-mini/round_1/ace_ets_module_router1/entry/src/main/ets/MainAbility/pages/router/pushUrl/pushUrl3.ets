
/*
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
import router from '@ohos.router';
import { BusinessError } from '@ohos.base';
@Entry
@Component
struct pushUrl3 {
  @State text1:string = ''

  build() {
    Column() {
      Text('pushUrl3').id('pushUrl3_text1')
      Text(this.text1).id('pushUrl3_text2')
      Button('pushUrl').onClick((event: ClickEvent) => {
        for(let i=0;i<33;i++){
          let text1 = this.text1;
          try {
            router.pushUrl({
              url: 'MainAbility/pages/router/pushUrl/pushUrl3'
            }).then()
              .catch((err:BusinessError) => {
                text1 = '' +  (err as BusinessError).code
              })
          } catch (err) {
            console.error(`pushUrl failed, code is ${(err as BusinessError).code}, message is ${(err as BusinessError).message}`);
          }
          this.text1 = text1
        }
      })
        .id('pushUrl3_btn1')
    }
  }
}