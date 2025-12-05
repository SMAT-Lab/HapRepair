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
struct pushUrl2 {

  build() {
    Column() {
      Text('pushUrl2').id('pushUrl2_text1')
      Button('pushUrl').onClick((event: ClickEvent) => {
        try {
          router.pushUrl({
            url: 'MainAbility/pages/router/pushUrl/page2'
          })
        } catch (err) {
          console.error(`pushUrl failed, code is ${(err as BusinessError).code}, message is ${(err as BusinessError).message}`);
        }
      })
        .id('pushUrl2_btn1')
    }
  }
}