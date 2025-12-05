
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
@Entry
@Component
struct Index {
build() {
Column() {
  CustomLayout({
    builder: columnChildren
  }).constraintSize({
    minWidth : 20,
    maxWidth: 50,
    minHeight : 10,
    maxHeight: 200})
}
.width('100%')
.height('100%')
.constraintSize({minWidth : 20, minHeight : 20})
}
}

@Builder
function columnChildren() {
  ForEach([1], (index: number) => {
    Text('S' + index)
      .fontSize(30)
      .width(100)
      .height(100)
      .constraintSize({minWidth : 100, maxHeight : 200})
  })
}

@Component
struct CustomLayout {
  @State height1:Length | undefined= 0
  @State height2:Length | undefined= 0
  @State width1:Length | undefined= 0
  @State width2:Length | undefined= 0
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
    console.log('llhh333 onPlaceChildren ' + constraint.minWidth + '-' + constraint.maxWidth + '-' + constraint.minHeight + '-' + constraint.maxHeight);
    console.log('llhh333 onPlaceChildren selfLayoutInfo ' + selfLayoutInfo.width + '-' + selfLayoutInfo.height);
    let startPos = 300;
    children.forEach((child, childIndex) => {
      let pos = startPos - child.measureResult.height;
      child.layout({ x: pos, y: pos })
    })
    if (constraint.maxHeight !== undefined && 
    constraint.minHeight !== undefined && 
    constraint.maxWidth !== undefined && 
    constraint.minWidth !== undefined){
      this.height1 = constraint.maxHeight
      this.height2 = constraint.minHeight
      this.width1 = constraint.maxWidth
      this.width2 = constraint.minWidth
    }

  }

  onMeasureSize(selfLayoutInfo: GeometryInfo, children: Array<Measurable>, constraint: ConstraintSizeOptions) {
    console.log('llhh333 onMeasureSize ' + constraint.minWidth + '-' + constraint.maxWidth + '-' + constraint.minHeight + '-' + constraint.maxHeight);
    console.log('llhh333 onMeasureSize selfLayoutInfo ' + selfLayoutInfo.width + '-' + selfLayoutInfo.height);
    constraint.minWidth = 20;
    constraint.minHeight = 30;
    console.log('llhh333 onMeasureSize ' + constraint.minWidth + '-' + constraint.maxWidth + '-' + constraint.minHeight + '-' + constraint.maxHeight);
    children.forEach((child, childIndex) => {
      child.measure(undefined)
    })
    this.result.width = 500;
    this.result.height = 600;
    return this.result;
  }

  build() {
    Column(){
      Text('' + this.height1).id('constraint_maxHeight1')
      Text('' + this.height2).id('constraint_minHeight1')
      Text('' + this.width1).id('constraint_maxWidth1')
      Text('' + this.width2).id('constraint_minWidth1')
      this.builder()
    }
  }
}