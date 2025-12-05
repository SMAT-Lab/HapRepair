/*
 * Copyright (c) 2023 Huawei Device Co., Ltd.
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

@Component
export struct AlphabetIndexerView {
  @Link _position: Position;
  private componentKey: string;
  private value: string[] = ['#', 'A', 'B', 'C', 'D', 'E', 'F', 'G',
    'H', 'I', 'J', 'K', 'L', 'M', 'N',
    'O', 'P', 'Q', 'R', 'S', 'T', 'U',
    'V', 'W', 'X', 'Y', 'Z']

  build() {
    AlphabetIndexer({ arrayValue: this.value, selected: 0 })
      .selectedColor(0xFFFFFF)
      .popupColor(0xFFFAF0)
      .selectedBackgroundColor(0xCCCCCC)
      .popupBackground(0xD2B48C)
      .usingPopup(true)
      .selectedFont({ size: 16, weight: FontWeight.Bolder })
      .popupFont({ size: 30, weight: FontWeight.Bolder })
      .itemSize(28)
      .alignStyle(IndexerAlign.Left)
      .position(this._position)
      .key(this.componentKey)
  }
}