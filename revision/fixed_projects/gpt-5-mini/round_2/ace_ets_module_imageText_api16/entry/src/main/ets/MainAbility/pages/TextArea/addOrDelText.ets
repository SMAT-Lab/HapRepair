
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
struct addOrDelTextDemo {
  controller1: TextAreaController = new TextAreaController();
  controller2: TextInputController = new TextInputController();
  controller3: SearchController = new SearchController();


  build() {
    Column() {
      Text().height('10%')
      TextArea({controller: this.controller1, text: '测试文本'})
        .height('10%')
        .id('textArea')

      TextInput({controller: this.controller2, text: '测试文本'})
        .id('textInput')
        .height('10%')

      Search({controller: this.controller3, value: '测试文本'})
        .id('search')
        .height('10%')

      Row(){
        Button('add')
          .id('add')
          .onClick(() => {
            this.controller1.addText('123')
            this.controller2.addText('123')
            this.controller3.addText('123')
          })
          .width('45%')

        Button('del')
          .id('del')
          .onClick(() => {
            this.controller1.deleteText({start: 0, end: 2})
            this.controller2.deleteText({start: 0})
            this.controller3.deleteText({end: 2})
          })
          .width('45%')
      }.height('10%')

      Row(){
        Button('add2')
          .id('add2')
          .width('45%')
          .onClick(() => {
            this.controller1.addText('添加文本',{offset:0})
            this.controller2.addText('添加文本',{offset:1})
            this.controller3.addText('添加文本',{offset:undefined})
          })

        Button('del2')
          .id('del2')
          .onClick(() => {
            this.controller1.deleteText({start: -1, end: -1})
            this.controller2.deleteText({start: -1, end: undefined})
            this.controller3.deleteText({start: undefined, end: -1})
          })
          .width('45%')
      }.height('10%')

      Row(){
        Button('del3')
          .id('del3')
          .onClick(() => {
            this.controller1.deleteText({start: 1.1, end: 2.3})
            this.controller2.deleteText({start: -1.1, end: -2.3})
            this.controller3.deleteText({start: null, end: -1})
          })
          .width('45%')

        Button('del4')
          .id('del4')
          .onClick(() => {
            this.controller1.deleteText({start: -1, end: -1})
            this.controller2.deleteText({start: -1, end: null})
            this.controller3.deleteText({start: null, end: -1})
          })
          .width('45%')
      }.height('10%')
      Row(){
        Button('del5')
          .id('del5')
          .onClick(() => {
            this.controller1.deleteText({start: undefined, end: -1})
            this.controller2.deleteText({start: undefined, end: 1})
            this.controller3.deleteText({start: null, end: 1})
          })
          .width('45%')

        Button('del6')
          .id('del6')
          .onClick(() => {
            this.controller1.deleteText({start: undefined, end: null})
            this.controller2.deleteText({start: 2, end: null})
            this.controller3.deleteText({start: 2, end: undefined})
          })
          .width('45%')
      }.height('10%')
      Row(){
        Button('del7')
          .id('del7')
          .onClick(() => {
            this.controller1.deleteText(undefined)
            this.controller2.deleteText(null)
            this.controller3.deleteText()
          })
          .width('45%')

      }.height('10%')
    }
    .height('100%')
  }

}