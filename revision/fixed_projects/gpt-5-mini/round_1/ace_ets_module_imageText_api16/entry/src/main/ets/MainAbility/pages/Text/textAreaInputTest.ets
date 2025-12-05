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

@Entry
@Component
struct TextAreaInputTestExample {
  @State text: string = '';

  build() {
    Column({ space: 20 }) {
      Column(){
        Text('这是一段测试文本这是一段测试文本')
          .id('textArea1')
          .draggable(true)
          .width('90%')
          .margin(20)
          .textAlign(TextAlign.Center)
          .copyOption(CopyOptions.LocalDevice)
          .onDragStart(() => {
            this.text = 'Text--onDragStart'
          })
          .onDragEnd(()=>{
            this.text = 'Text--onDragEnd'
          })

        TextArea({ placeholder: this.text })
          .id('textArea2')
          .width(336)
          .height(56)
          .margin(20)
          .copyOption(CopyOptions.LocalDevice)


      }.width('100%').height('100%')
    }
  }
}