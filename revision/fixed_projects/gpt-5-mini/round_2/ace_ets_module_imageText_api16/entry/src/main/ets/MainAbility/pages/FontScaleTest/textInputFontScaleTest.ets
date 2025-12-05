
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
struct textInputFontScaleTest {
  @State text: string = 'As the sun begins to set, casting a warm golden hue across the sky'
  build() {
    Row() {
        TextInput({ text: this.text})
          .maxFontScale(undefined)
          .minFontScale(undefined)
          .id('input1')
          .height('5%')
        TextInput({ text: this.text})
          .maxFontScale($r('app.string.maxFontScale'))
          .minFontScale($r('app.string.minFontScale'))
          .id('input2')
          .height('5%')
        TextInput({ text: this.text})
          .maxFontScale(2)
          .minFontScale(0.5)
          .id('input3')
          .height('5%')
        TextInput({ text: this.text})
          .maxFontScale(0.5)
          .minFontScale(-1)
          .id('input4')
          .height('5%')
        TextInput({ text: this.text})
          .maxFontScale(0.5)
          .minFontScale(2)
          .id('input5')
          .height('5%')
		TextInput({ text: this.text})
          .maxFontScale($r('app.boolean.maxFontScale'))
          .minFontScale($r('app.boolean.minFontScale'))
          .id('input6')
		  .height('5%')
        TextInput({ text: this.text})
          .maxFontScale($r('app.color.maxFontScale'))
          .minFontScale($r('app.color.minFontScale'))
          .id('input7')
		  .height('5%')
        TextInput({ text: this.text})
          .maxFontScale($r('app.float.maxFontScale'))
          .minFontScale($r('app.float.minFontScale'))
          .id('input8')
		  .height('5%')
        TextInput({ text: this.text})
          .maxFontScale($r('app.integer.maxFontScale'))
          .minFontScale($r('app.integer.minFontScale'))
          .id('input9')
		  .height('5%')
        TextInput({ text: this.text})
          .maxFontScale($r('app.intarray.maxFontScale'))
          .minFontScale($r('app.intarray.minFontScale'))
          .id('input10')
		  .height('5%')
        TextInput({ text: this.text})
          .maxFontScale($r('app.pattern.maxFontScale'))
          .minFontScale($r('app.pattern.minFontScale'))
          .id('input11')
		  .height('5%')
        TextInput({ text: this.text})
          .maxFontScale($r('app.plural.maxFontScale'))
          .minFontScale($r('app.plural.minFontScale'))
          .id('input12')
		  .height('5%')
        TextInput({ text: this.text})
          .maxFontScale($r('app.strarray.maxFontScale'))
          .minFontScale($r('app.strarray.minFontScale'))
          .id('input13')
		  .height('5%')
    }
  }
}