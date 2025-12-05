
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

@Entry
@Component
struct textInputHalfLeadingTest {
  @State text: string = 'As the sun begins to set, casting a warm golden hue across the sky'
  build() {
    Row() {
      Column({ space: 20 }) {
        TextInput({ text:this.text})
          .halfLeading(true)
          .id('input1')
          .height('10%')
        TextInput({ text: this.text})
          .halfLeading(false)
          .id('input2')
          .height('10%')
        TextInput({ text: this.text})
          .halfLeading(undefined)
          .id('input3')
          .height('10%')
        TextInput({ text: this.text})
          .halfLeading(null)
          .id('input4')
          .height('10%')
      }
      .height('100%')
    }
  }
}