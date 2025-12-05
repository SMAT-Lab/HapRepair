
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
struct textAreaEllipsisModeTest {
  @State text: string = 'As the sun begins to set, casting a warm golden hue across the sky,' +
    'the world seems to slow down and breathe a sigh of relief. The sky is painted with hues of orange, ' +
    ' pink, and lavender, creating a breath taking tapestry that stretches as far as the eye can see.' +
    'The air is filled with the sweet scent of blooming flowers, mingling with the earthy aroma of freshly turned soil.'
  build() {
    Row() {
      Column({ space: 20 }) {
        TextArea({ text: this.text})
          .textOverflow(TextOverflow.Ellipsis)
          .ellipsisMode(EllipsisMode.END)
          .maxLines(1)
          .id('area1')
          .height('10%')
        TextArea({ text: this.text})
          .textOverflow(TextOverflow.Ellipsis)
          .ellipsisMode(EllipsisMode.START)
          .maxLines(1)
          .id('area2')
          .height('10%')
        TextArea({ text: this.text})
          .textOverflow(TextOverflow.Ellipsis)
          .ellipsisMode(EllipsisMode.CENTER)
          .maxLines(1)
          .id('area3')
          .height('10%')
        TextArea({ text: this.text})
          .textOverflow(TextOverflow.Ellipsis)
          .ellipsisMode(undefined)
          .maxLines(1)
          .id('area4')
          .height('10%')
        TextArea({ text: this.text})
          .textOverflow(TextOverflow.Ellipsis)
          .ellipsisMode(null)
          .maxLines(1)
          .id('area5')
          .height('10%')
      }
    }
  }
}