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
struct pushUrl6 {

  build() {
    Column() {
      Text('pushUrl6').id('pushUrl6_text1')
      Button('pushUrl').onClick((event: ClickEvent) => {
        router.pushUrl({
          url: 'MainAbility/pages/router/pushUrl/page2',
        }, (err) => {
          if (err) {
            console.error(`pushUrl failed, code is ${err.code}, message is ${err.message}`);
            return;
          }
          console.info('pushUrl success');
        })
      })
        .id('pushUrl6_btn1')
    }
  }
}