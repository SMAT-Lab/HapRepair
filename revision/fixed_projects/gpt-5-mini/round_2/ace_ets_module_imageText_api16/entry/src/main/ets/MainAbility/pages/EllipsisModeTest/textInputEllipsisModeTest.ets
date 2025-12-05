
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
struct textInputEllipsisModeTest {
  @State text: string = 'As the sun begins to set, casting a warm golden hue across the sky,' +
    'the world seems to slow down and breathe a sigh of relief. The sky is painted with hues of orange, ' +
    ' pink, and lavender, creating a breath taking tapestry that stretches as far as the eye can see.' +
    'The air is filled with the sweet scent of blooming flowers, mingling with the earthy aroma of freshly turned soil.'
  build() {
      Column({ space: 20 }) {
        TextInput({ text: this.text})
          .textOverflow(TextOverflow.Ellipsis)
          .ellipsisMode(EllipsisMode.END)
          .style(TextInputStyle.Inline)
          .id('input1')
          .height('10%')
        TextInput({ text: this.text})
          .textOverflow(TextOverflow.Ellipsis)
          .ellipsisMode(EllipsisMode.START)
          .style(TextInputStyle.Inline)
          .id('input2')
          .height('10%')
        TextInput({ text: this.text})
          .textOverflow(TextOverflow.Ellipsis)
          .ellipsisMode(EllipsisMode.CENTER)
          .style(TextInputStyle.Inline)
          .id('input3')
          .height('10%')
        TextInput({ text: this.text})
          .textOverflow(TextOverflow.Ellipsis)
          .ellipsisMode(undefined)
          .style(TextInputStyle.Inline)
          .id('input4')
          .height('10%')
        TextInput({ text: this.text})
          .textOverflow(TextOverflow.Ellipsis)
          .ellipsisMode(null)
          .style(TextInputStyle.Inline)
          .id('input5')
          .height('10%')
      }
    }
  }
}