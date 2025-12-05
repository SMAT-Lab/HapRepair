
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

import { AlertDialog, PopoverDialog, PopoverOptions, SymbolGlyphModifier } from '@kit.ArkUI';
@Entry
@Component
struct Index {
  @State color:Color = Color.Pink
  controller1: TextController = new TextController();
  styledString2: StyledString = new StyledString('运动45分钟');
  mutableStyledString2: MutableStyledString = new MutableStyledString('test hello world', [{
    start: 0,
    length: 5,
    styledKey: StyledStringKey.FONT,
    styledValue: new TextStyle({ fontColor: Color.Blue })
  }]);

  searchController: SearchController = new SearchController()

  async onPageShow() {
    this.controller1.setStyledString(this.mutableStyledString2)
  }

  build() {
    Column({space:20}) {
      Column() {
        Text('禁掉onclick')
        Hyperlink('www.baidu.com')
          .id('hyperlink1')
          .color(this.color)// .color(Color.Pink)
          .onClick((event) => {
            console.log('Hyperlink1 Click')
            try {
              console.log('Hyperlink1 Click Begin')
              event.preventDefault();
              console.log('Hyperlink1 Click After')
            } catch (e) {
              console.log('Hyperlink1 Click ErrorCode:' + JSON.stringify(e))
              console.log('Hyperlink1 Click ErrorMessage:' + e.message)
            }
          })
        Text('不禁掉onclick')
        Hyperlink('www.baidu.com')
          .id('hyperlink2')
          .color(this.color)// .color(Color.Pink)

        Text('禁掉onTouch')
        Hyperlink('www.baidu.com')
          .id('hyperlink3')
          .onTouch((event) => {
            console.log('Hyperlink1 Touch')
            try {
              console.log('Hyperlink1 Touch Begin')
              event.preventDefault();
              console.log('Hyperlink1 Touch After')
            } catch (e) {
              console.log('Hyperlink1 Touch ErrorCode:' + JSON.stringify(e))
              console.log('Hyperlink1 Touch ErrorMessage:' + e.message)
            }
          })

      }
    }
    .borderWidth(1)
    .borderColor(Color.Red)
    .width('100%')
    .height('100%')
  }
}