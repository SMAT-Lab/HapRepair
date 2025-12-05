
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
struct searchButtonDemo {
  color:Color = Color.Pink
  isShow: boolean = false
  isShow2: boolean = false
  sheetHeight: number = 300;
  isShowdialog: boolean = false
  handlePopup: boolean = false
  customPopup: boolean = false
  text1: string = ''
  text2: string = ''
  controller1: TextController = new TextController();
  styledString2: StyledString = new StyledString('运动45分钟');
  mutableStyledString2: MutableStyledString = new MutableStyledString('test hello world', [{
    start: 0,
    length: 5,
    styledKey: StyledStringKey.FONT,
    styledValue: new TextStyle({ fontColor: Color.Blue })
  }]);

  searchController: SearchController = new SearchController()
  changeValue: string = ''
  submitValue: string = ''

  async onPageShow() {
    this.controller1.setStyledString(this.mutableStyledString2)
  }

  build() {
    Column({space:20}) {
      Column() {
        Search({ value: this.changeValue, placeholder: 'Type to search...', controller: this.searchController })
          .id('search1')
          .searchIcon(new SymbolGlyphModifier($r('sys.symbol.magnifyingglass')).fontColor([Color.Red]))
          .cancelButton({
            style: CancelButtonStyle.CONSTANT,
            icon: new SymbolGlyphModifier($r('sys.symbol.xmark')).fontColor([Color.Green])
          })
          .searchButton('SEARCH', {
            fontSize: '16fp',
            fontColor: '#ff3f97e9',
            autoDisable: true
          })
          .width('95%')
          .height(40)
          .backgroundColor('#F5F5F5')
          .placeholderColor(Color.Grey)
          .placeholderFont({ size: 14, weight: 400 })
          .textFont({ size: 14, weight: 400 })
          .margin(10)
          .onClick(() => {
            console.log('preventDefault')
          })

        Search({ value: this.changeValue, placeholder: 'Type to search...', controller: this.searchController })
          .id('search2')
          .searchIcon(new SymbolGlyphModifier($r('sys.symbol.magnifyingglass')).fontColor([Color.Red]))
          .cancelButton({
            style: CancelButtonStyle.CONSTANT,
            icon: new SymbolGlyphModifier($r('sys.symbol.xmark')).fontColor([Color.Green])
          })
          .searchButton('SEARCH', {
            fontSize: '16fp',
            fontColor: '#ff3f97e9',
            autoDisable: false
          })
          .width('95%')
          .height(40)
          .backgroundColor('#F5F5F5')
          .placeholderColor(Color.Grey)
          .placeholderFont({ size: 14, weight: 400 })
          .textFont({ size: 14, weight: 400 })
          .margin(10)
          .onClick(() => {
            console.log('preventDefault222')
          })

        Search({ value: this.changeValue, placeholder: 'Type to search...', controller: this.searchController })
          .id('search3')
          .searchIcon(new SymbolGlyphModifier($r('sys.symbol.magnifyingglass')).fontColor([Color.Red]))
          .cancelButton({
            style: CancelButtonStyle.CONSTANT,
            icon: new SymbolGlyphModifier($r('sys.symbol.xmark')).fontColor([Color.Green])
          })
          .searchButton('SEARCH', {
            fontSize: '16fp',
            fontColor: '#ff3f97e9',
            autoDisable: undefined
          })
          .width('95%')
          .height(40)
          .backgroundColor('#F5F5F5')
          .placeholderColor(Color.Grey)
          .placeholderFont({ size: 14, weight: 400 })
          .textFont({ size: 14, weight: 400 })
          .margin(10)
          .onClick(() => {
            console.log('preventDefault222')
          })

      }
    }
    .borderWidth(1)
    .borderColor(Color.Red)
    .width('100%')
    .height('100%')
  }
}