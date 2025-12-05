/**
 * Copyright (c) 2025 Huawei Device Co., Ltd.
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
struct TextInputStopBackPress {

  build() {
    Column({ space: 1 }) {
      Text('TextInput:stopBackPress').height('10%')
      TextInput({ placeholder: 'StopBackPress: true' })
        .id("TextInput_stopBackPress_true")
        .stopBackPress(true)
        .height('20%')

      TextInput({ placeholder: 'StopBackPress: false' })
        .id("TextInput_stopBackPress_false")
        .stopBackPress(false)
        .height('20%')

      TextInput({ placeholder: 'StopBackPress: default' })
        .id("TextInput_stopBackPress_default")
        .height('20%')

      TextInput({ placeholder: 'StopBackPress: undefined' })
        .id("TextInput_stopBackPress_undefined")
        .stopBackPress(undefined)
        .height('20%')
    }.width('100%')
  }
}