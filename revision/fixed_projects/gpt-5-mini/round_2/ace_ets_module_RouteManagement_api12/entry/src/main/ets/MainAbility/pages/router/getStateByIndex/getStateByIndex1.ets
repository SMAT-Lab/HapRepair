

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
struct getStateByIndex1 {
  @State str1:string = '1'
  str2:string = '2'
  str3:string = '3'
  str4:string = '4'


  build() {
    Column({space:5}) {

      Text(this.str1).id('getStateByIndex1_text1')
      Button('getStateByIndex(33)')
        .id('getStateByIndex1_btn1')
        .onClick(()=>{
          this.str1 = String(router.getStateByIndex(33))
        })

    }.width('100%').height('100%')


  }
}