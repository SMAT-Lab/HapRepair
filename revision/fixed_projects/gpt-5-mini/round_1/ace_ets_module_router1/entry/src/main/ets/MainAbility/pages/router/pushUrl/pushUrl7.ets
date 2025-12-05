
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
@Entry
@Component
struct pushUrl7 {
  @State text1:string = ''

  build() {
    Column() {
      Text('pushUrl7').id('pushUrl7_text1')
      Text(this.text1).id('pushUrl7_text2')
      Button('pushUrl').onClick((event: ClickEvent) => {
        let text1 = this.text1
        for(let i = 0; i<=33 ; i++){
          router.pushUrl({
            url: 'MainAbility/pages/router/pushUrl/pushUrl7',
          }, (err) => {
            if (err) {
              text1 = '' +  err.code
              return;
            }
            console.info('pushUrl success');
          })
        }
        this.text1 = text1
      })
        .id('pushUrl7_btn1')
    }
  }
}