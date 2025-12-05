
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
import matrix4 from '@ohos.matrix4'

let matrix = matrix4.init([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0])

@Entry
@Component
struct comSizeAbilityTest_26 {
  controller: RichEditorController = new RichEditorController();
  options: RichEditorOptions = { controller: this.controller };
  private start: number = -1;
  private end: number = -1;
  @State message: string = "[-1, -1]"
  // span 未发生变化，改为普通变量
  span: string = "文本组件"

  build() {
    Scroll() {
      Column() {
        Row() {
          Button("更新样式:加粗").onClick(() => {
            this.controller.updateSpanStyle({
              start: this.start,
              end: this.end,
              textStyle:
              {
                fontWeight: FontWeight.Bolder
              }
            })
          })
        }

        Column() {
          RichEditor(this.options)
            .size({ width: 100, height: 100 })
            .offset({ x: 1, y: 1 })
            .translate({ x: 1, y: 1, z: 1 })
            .scale({
              x: 1,
              y: 1,
              z: 1,
              centerX: 1,
              centerY: 1
            })
            .rotate({
              x: 1,
              y: 1,
              z: 1,
              angle: 0,
              centerX: 1,
              centerY: 1
            })
            .transform(matrix)
            .backgroundColor('blue')
            .onReady(() => {
              this.controller.addTextSpan(this.span,
                {
                  style:
                  {
                    fontColor: Color.Orange,
                    fontSize: 12
                  }
                })
            })
            .key('RichEditorSpan_localOffset')
            .onSelect((value: RichEditorSelection) => {
              this.start = value.selection[0];
              this.end = value.selection[1];
              this.message = "[" + this.start + ", " + this.end + "]"
            })
            .aboutToIMEInput((value: RichEditorInsertValue) => {
              console.log("---------------------- aboutToIMEInput ----------------------")
              console.log("insertOffset:" + value.insertOffset)
              console.log("insertValue:" + value.insertValue)
              return true;
            })
            .onIMEInputComplete((value: RichEditorTextSpanResult) => {
              console.log("---------------------- onIMEInputComplete ---------------------")
              console.log("spanIndex:" + value.spanPosition.spanIndex)
              console.log("spanRange:[" + value.spanPosition.spanRange[0] + "," + value.spanPosition.spanRange[1] + "]")
              console.log("offsetInSpan:[" + value.offsetInSpan[0] + "," + value.offsetInSpan[1] + "]")
              console.log("value:" + value.value)
            })
            .aboutToDelete((value: RichEditorDeleteValue) => {
              console.log("---------------------- aboutToDelete --------------------------")
              console.log("offset:" + value.offset)
              console.log("direction:" + value.direction)
              console.log("length:" + value.length)
              value.richEditorDeleteSpans.forEach(item => {
                console.log("---------------------- item --------------------------")
                console.log("spanIndex:" + item.spanPosition.spanIndex)
                console.log("spanRange:[" + item.spanPosition.spanRange[0] + "," + item.spanPosition.spanRange[1] + "]")
                console.log("offsetInSpan:[" + item.offsetInSpan[0] + "," + item.offsetInSpan[1] + "]")
                if (typeof (item as RichEditorImageSpanResult)['imageStyle'] != 'undefined') {
                  console.log("image:" + (item as RichEditorImageSpanResult).valueResourceStr)
                } else {
                  console.log("text:" + (item as RichEditorTextSpanResult).value)
                }
                if (typeof (item as RichEditorImageSpanResult)['imageStyle'] != 'undefined') {
                  console.log("image:" + (item as RichEditorImageSpanResult).valueResourceStr)
                } else {
                  console.log("text:" + (item as RichEditorTextSpanResult).value)
                }
              })
              return true;
            })
            .onDeleteComplete(() => {
              console.log("---------------------- onDeleteComplete ------------------------")
            })
            .borderWidth(1)
            .borderColor(Color.Green)
        }
        .key('RichEditor_Father')
        .borderWidth(1)
        .borderColor(Color.Red)
        .width("100%")
        .height("70%")
      }
    }.height('100%')
  }
}