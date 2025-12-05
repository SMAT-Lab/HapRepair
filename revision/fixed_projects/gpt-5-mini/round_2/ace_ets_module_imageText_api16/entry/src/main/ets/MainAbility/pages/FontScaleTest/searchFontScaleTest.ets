
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
struct searchFontScaleTest {
  // 直接使用一般变量即可
  text: string = 'As the sun begins to set, casting a warm golden hue across the sky'
  build() {
    Column() {
      Column() {
        Search({ value:this.text})
          .maxFontScale(undefined)
          .minFontScale(undefined)
          .id('search1')
          .height('5%')
          .margin(0)
        Search({ value: this.text})
          .maxFontScale($r('app.string.maxFontScale'))
          .minFontScale($r('app.string.minFontScale'))
          .id('search2')
          .height('5%')
          .margin(0)
        Search({ value: this.text})
          .maxFontScale(2)
          .minFontScale(0.5)
          .id('search3')
          .height('5%')
          .margin(0)
        Search({ value: this.text})
          .maxFontScale(0.5)
          .minFontScale(-1)
          .id('search4')
          .height('5%')
          .margin(0)
        Search({ value: this.text})
          .maxFontScale(0.5)
          .minFontScale(2)
          .id('search5')
          .height('5%')
          .margin(0)
		Search({ value: this.text})
          .maxFontScale($r('app.boolean.maxFontScale'))
          .minFontScale($r('app.boolean.minFontScale'))
          .id('search6')
		  .height('5%')
          .margin(0)
        Search({ value: this.text})
          .maxFontScale($r('app.color.maxFontScale'))
          .minFontScale($r('app.color.minFontScale'))
          .id('search7')
		  .height('5%')
          .margin(0)
        Search({ value: this.text})
          .maxFontScale($r('app.float.maxFontScale'))
          .minFontScale($r('app.float.minFontScale'))
          .id('search8')
		  .height('5%')
          .margin(0)
        Search({ value: this.text})
          .maxFontScale($r('app.integer.maxFontScale'))
          .minFontScale($r('app.integer.minFontScale'))
          .id('search9')
		  .height('5%')
          .margin(0)
        Search({ value: this.text})
          .maxFontScale($r('app.intarray.maxFontScale'))
          .minFontScale($r('app.intarray.minFontScale'))
          .id('search10')
		  .height('5%')
          .margin(0)
        Search({ value: this.text})
          .maxFontScale($r('app.pattern.maxFontScale'))
          .minFontScale($r('app.pattern.minFontScale'))
          .id('search11')
		  .height('5%')
          .margin(0)
        Search({ value: this.text})
          .maxFontScale($r('app.plural.maxFontScale'))
          .minFontScale($r('app.plural.minFontScale'))
          .id('search12')
		  .height('5%')
          .margin(0)
        Search({ value: this.text})
          .maxFontScale($r('app.strarray.maxFontScale'))
          .minFontScale($r('app.strarray.minFontScale'))
          .id('search13')
		  .height('5%')
          .margin(0)
      }
    }
    .height('100%')
  }
}