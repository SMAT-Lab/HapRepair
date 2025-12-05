/*
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
struct TextOnWillChangeOptions {
  @State oldContent: string = ''
  @State newContent: string = ''
  @State oldPreviewTextValue: string = ''
  @State oldPreviewTextOffset: string = ''
  @State newPreviewTextValue: string = ''
  @State newPreviewTextOffset: string = ''
  @State rangeBefore: string = ''
  @State rangeAfter: string = ''
  controller1: SearchController  = new SearchController();
  controller2: TextAreaController = new TextAreaController();
  controller3: TextInputController = new TextInputController();
  build() {
    Column() {
      Row(){
        Text(this.oldContent).id('oldContent').textAlign(TextAlign.Center).width('45%')
        Text(this.newContent).id('newContent').textAlign(TextAlign.Center).width('45%')
      }.height('5%')
      Row(){
        Text(this.oldPreviewTextValue).id('oldPreviewTextValue').textAlign(TextAlign.Center).width('45%')
        Text(this.oldPreviewTextOffset).id('oldPreviewTextOffset').textAlign(TextAlign.Center).width('45%')
      }.height('5%')
      Row(){
        Text(this.newPreviewTextValue).id('newPreviewTextValue').textAlign(TextAlign.Center).width('45%')
        Text(this.newPreviewTextOffset).id('newPreviewTextOffset').textAlign(TextAlign.Center).width('45%')
      }.height('5%')
      Row(){
        Text(this.rangeBefore).id('rangeBefore').textAlign(TextAlign.Center).width('45%')
        Text(this.rangeAfter).id('rangeAfter').textAlign(TextAlign.Center).width('45%')
      }.height('5%')


      Search({value: '12',controller:this.controller1})
        .id('search').height('5%')
        .onWillChange((value: EditableTextChangeValue) => {
          this.oldContent = '' + value.options?.oldContent;
          this.newContent = '' + value.content;
          this.oldPreviewTextValue = '' + value.options?.oldPreviewText?.value;
          this.oldPreviewTextOffset = '' + value.options?.oldPreviewText?.offset;
          this.newPreviewTextValue = '' + value.previewText?.value;
          this.newPreviewTextOffset = '' + value.previewText?.offset;
          this.rangeBefore = '' + value.options?.rangeBefore.start + '-' + value.options?.rangeBefore.end;
          this.rangeAfter = '' + value.options?.rangeAfter.start + '-' + value.options?.rangeAfter.end;
          return true;
        })

      TextArea({text: '12',controller:this.controller2})
        .id('textArea').height('5%')
        .onWillChange((value: EditableTextChangeValue) => {
          this.oldContent = '' + value.options?.oldContent;
          this.newContent = '' + value.content;
          this.oldPreviewTextValue = '' + value.options?.oldPreviewText?.value;
          this.oldPreviewTextOffset = '' + value.options?.oldPreviewText?.offset;
          this.newPreviewTextValue = '' + value.previewText?.value;
          this.newPreviewTextOffset = '' + value.previewText?.offset;
          this.rangeBefore = '' + value.options?.rangeBefore.start + '-' + value.options?.rangeBefore.end;
          this.rangeAfter = '' + value.options?.rangeAfter.start + '-' + value.options?.rangeAfter.end;
          return true;
        })

      TextInput({text: '12',controller:this.controller3})
        .id('textInput').height('5%')
        .onWillChange((value: EditableTextChangeValue) => {
          this.oldContent = '' + value.options?.oldContent;
          this.newContent = '' + value.content;
          this.oldPreviewTextValue = '' + value.options?.oldPreviewText?.value;
          this.oldPreviewTextOffset = '' + value.options?.oldPreviewText?.offset;
          this.newPreviewTextValue = '' + value.previewText?.value;
          this.newPreviewTextOffset = '' + value.previewText?.offset;
          this.rangeBefore = '' + value.options?.rangeBefore.start + '-' + value.options?.rangeBefore.end;
          this.rangeAfter = '' + value.options?.rangeAfter.start + '-' + value.options?.rangeAfter.end;
          return true;
        })
    }
    .width('100%')
    .height('100%')
  }
}
