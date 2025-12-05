/**
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
import {MessageManager,Callback} from '../../common/MessageManager';

@Entry
@Component
struct onMeasureSize2 {
  onPageShow() {
    console.info('onMeasureSize  onPageShow');
  }

  build() {
    Column() {
      CustomLayout({ builder: ColumnChildren })
    }.key('onMeasureSize2_Column_1')
  }
}

@Builder
function ColumnChildren() {
  Text('S1')
    .fontSize(30)
    .width(100)
    .height(100)
    .borderWidth(2)
    .key("onMeasureSize2_text_01")

  Text('S2')
    .fontSize(30)
    .width(100)
    .height(100)
    .borderWidth(2)
    .key("onMeasureSize2_text_02")

  Text('S3')
    .fontSize(30)
    .width(300)
    .height(300)
    .borderWidth(2)
    .key("onMeasureSize2_text_03")

}

@Component
struct CustomLayout {
  @Builder
  doNothingBuilder() {
  };

  @BuilderParam builder: () => void = this.doNothingBuilder;
  startSize: number = 100;
  result: SizeResult = {
    width: 0,
    height: 0
  };

  onPlaceChildren(selfLayoutInfo: GeometryInfo, children: Array<Layoutable>, constraint: ConstraintSizeOptions) {
    let startPos = 300;
    children.forEach((child, child_index) => {
      if (child_index != 2) {
        let pos = startPos - child.measureResult.height;
        child.layout({ x: pos, y: pos })

      }
    })
  }

  onMeasureSize(selfLayoutInfo: GeometryInfo, children: Array<Measurable>, constraint: ConstraintSizeOptions) {
    let size = 100;
    children.forEach((child, child_index) => {
      if (child_index != 2) {
        let result: MeasureResult = child.measure({
          minHeight: size,
          minWidth: size,
          maxWidth: size,
          maxHeight: size
        })
        size += result.width / 2;
      }
    })
    this.result.width = 100;
    this.result.height = 400;
    return this.result;
  }

  build() {
    this.builder()
  }
}