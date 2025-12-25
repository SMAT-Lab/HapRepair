


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
import events_emitter from '@ohos.events.emitter';

class MyDataSourceSwiper implements IDataSource {
  private list: number[] = []

  constructor(list: number[]) {
    this.list = list
  }

  totalCount(): number {
    return this.list.length
  }

  getData(index: number): number {
    return this.list[index]
  }

  registerDataChangeListener(listener: DataChangeListener): void {
  }

  unregisterDataChangeListener() {
  }
}

@Entry
@Component
struct SwiperExample3 {
  private swiperController: SwiperController = new SwiperController()
  private data: MyDataSourceSwiper = new MyDataSourceSwiper([])
  isHoverShow: boolean = false
  isTrue: boolean = true
  isArrow: boolean = true
  arrowStyle: ArrowStyle = {}
  boolFlag: boolean = false
  bool: boolean = false
  boolIsNum: boolean = false
  boolIsStr: boolean = false
  boolIsNull: boolean = false
  boolNum: number = 2
  boolStr: string = ''
  indicator: boolean = true
  private colorList: ESObject = [
    { cardId: 1, color: Color.White },
    { cardId: 2, color: Color.Yellow },
    { cardId: 3, color: Color.Blue },
    { cardId: 4, color: Color.Brown },
    { cardId: 5, color: Color.Gray },
    { cardId: 6, color: Color.Green },
    { cardId: 7, color: Color.Grey },
    { cardId: 8, color: Color.Orange },
    { cardId: 9, color: Color.Pink },
    { cardId: 10, color: Color.Red }]
  private valueChangeCallBack = (eventData: events_emitter.EventData) => {
    console.info("key==> page valueChangeCallBack");
    console.info("valueChangeCallBack eventData:" + JSON.stringify(eventData));
    console.info("valueChangeCallBack eventData.data?.type:" + JSON.stringify(eventData.data?.type));
    if (eventData != null) {
      if (eventData.data?.type == undefined) {
        this.arrowStyle = JSON.parse(eventData.data?.arrowStyle)
      }
      if (eventData.data?.type == "BoolTrue") {
        console.info("valueChangeCallBack BoolTrue:" + JSON.stringify(eventData));
        this.boolFlag = true
        this.bool = true
        console.info("valueChangeCallBack this.bool:" + JSON.stringify(this.bool));
      }
      if (eventData.data?.type == "BoolFalse") {
        console.info("valueChangeCallBack BoolFalse:" + JSON.stringify(eventData));
        this.boolFlag = true
        this.bool = false
      }
      if (eventData.data?.type == "BoolErrNum") {
        console.info("valueChangeCallBack BoolErrNum:" + JSON.stringify(eventData));
        this.boolFlag = false
        this.boolIsNum = true
        this.boolNum = Number(eventData.data?.displayArrow)
      }
      if (eventData.data?.type == "BoolErrStr") {
        console.info("valueChangeCallBack BoolErrStr:" + JSON.stringify(eventData));
        this.boolFlag = false
        this.boolIsNum = false
        this.boolIsStr = true
        this.boolStr = eventData.data?.displayArrow
      }
      if (eventData.data?.type == "BoolErrNull") {
        console.info("valueChangeCallBack BoolErrNull:" + JSON.stringify(eventData));
        this.boolFlag = false
        this.boolIsNum = false
        this.boolIsStr = false
        this.boolIsNull = true
      }
      if (eventData.data?.type == "BoolErrUndefined") {
        console.info("valueChangeCallBack BoolErrUndefined:" + JSON.stringify(eventData));
        this.boolFlag = false
        this.boolIsNum = false
        this.boolIsStr = false
        this.boolIsNull = false
      }
    }
  }

  aboutToAppear(): void {
    let valueChangeEvent: events_emitter.InnerEvent = {
      eventId: 10,
      priority: events_emitter.EventPriority.LOW
    }
    events_emitter.on(valueChangeEvent, this.valueChangeCallBack)
  }

  build() {
    Column({ space: 5 }) {
      Column() {
        Swiper(this.swiperController) {
          ForEach(this.colorList, (item: ESObject, index) => {
            Text("第" + (item.cardId).toString() + "页")
              .width('90%')
              .height('15%')
              .backgroundColor(item.color)
              .textAlign(TextAlign.Center)
              .fontSize(20)
          })
        }
        .key("SwiperA")
        .backgroundColor($r("sys.color.ohos_id_color_component_normal"))
        .foregroundColor($r("sys.color.ohos_id_color_primary"))
        .index(0)
        .indicator(this.indicator)
        .cachedCount(1)
        .loop(false)
        .duration(1000)
        .vertical(false)
        .onChange((index: number) => {
          console.info(index.toString())
        })
        .displayArrow(this.arrowStyle, true)
      }

      Column() {
        Swiper(this.swiperController) {
          ForEach(this.colorList, (item: ESObject, index) => {
            Text("第" + (item.cardId).toString() + "页")
              .width('90%')
              .height('15%')
              .backgroundColor(item.color)
              .textAlign(TextAlign.Center)
              .fontSize(20)
          })
        }
        .key("SwiperB")
        .index(0)
        .indicator(this.indicator)
        .cachedCount(1)
        .loop(false)
        .duration(1000)
        .vertical(false)
        .onChange((index: number) => {
          console.info(index.toString())
        })
        .displayArrow((this.boolFlag ? this.bool : (this.boolIsNum ? !!this.boolNum :
          (this.boolIsStr ? (this.boolStr == "true") : (this.boolIsNull ? null : undefined)))), true)
      }

      Column() {
        Swiper(this.swiperController) {
          ForEach(this.colorList, (item: ESObject, index) => {
            Text("第" + (item.cardId).toString() + "页")
              .width('90%')
              .height('15%')
              .backgroundColor(item.color)
              .textAlign(TextAlign.Center)
              .fontSize(20)
          })
        }
        .key("SwiperC")
        .index(0)
        .indicator(this.indicator)
        .cachedCount(1)
        .loop(false)
        .duration(1000)
        .vertical(false)
        .onChange((index: number) => {
          console.info(index.toString())
        })
        .displayArrow(true, (this.boolFlag ? this.bool : (this.boolIsNum ? !!this.boolNum :
          (this.boolIsStr ? (this.boolStr == "true") : (this.boolIsNull ? null : undefined)))))
      }

      Column() {
        Swiper(this.swiperController) {
          ForEach(this.colorList, (item: ESObject, index) => {
            Text("第" + (item.cardId).toString() + "页")
              .width('90%')
              .height('15%')
              .backgroundColor(item.color)
              .textAlign(TextAlign.Center)
              .fontSize(20)
          })
        }
        .key("SwiperC")
        .index(0)
        .indicator(this.indicator)
        .cachedCount(1)
        .loop(false)
        .duration(1000)
        .vertical(false)
        .onChange((index: number) => {
          console.info(index.toString())
        })
        .displayArrow(true, (this.boolFlag ? this.bool : (this.boolIsNum ? !!this.boolNum :
          (this.boolIsStr ? (this.boolStr == "true") : (this.boolIsNull ? null : undefined)))))

      }

@Reusable
@Component
struct SwiperDataItemComponent {
  item: string = ''
  build() {
    Text(this.item)
      .width('90%')
      .height('15%')
      .backgroundColor(0xAFEEEE)
      .textAlign(TextAlign.Center)
      .fontSize(20)
  }
}

      Column() {
        Swiper(this.swiperController) {
          LazyForEach(this.data, (item: string, index) => {
            SwiperDataItemComponent({ item: item })
          }, (item: string) => item)
        }
        .key("SwiperD")
        .foregroundColor($r("sys.color.ohos_id_color_primary"))
        .index(0)
        .indicator(this.indicator) //是否启用导航点指示器。
        .cachedCount(1)
        .loop(false)
        .duration(1000)
        .vertical(false)
        .onChange((index: number) => {
          console.info(index.toString())
        })
        .displayArrow(this.arrowStyle)
      }

      Column() {
        Swiper(this.swiperController) {
          LazyForEach(this.data, (item: string) => {
            SwiperTextItemComponent({
              text: item
            }).reuseId(item)
          }, (item: string) => item)
        }
        .key("SwiperE")
        .foregroundColor($r("sys.color.ohos_id_color_primary"))
        .index(0)
        .indicator(this.indicator)
        .cachedCount(1)
        .loop(false)
        .duration(1000)
        .vertical(false)
        .onChange((index: number) => {
          console.info(index.toString())
        })
        .displayArrow(true)
      }

@Reusable
@Component
struct SwiperTextItemComponent {
  text: string = ''
  build() {
    Text(this.text)
      .width('90%')
      .height('15%')
      .backgroundColor(0xAFEEEE)
      .textAlign(TextAlign.Center)
      .fontSize(20)
  }
}

      Column() {
        Swiper(this.swiperController) {
          LazyForEach(this.data, (item: string, index) => {
            SwiperDataItemComponent({ item: item }).reuseId(item)
          }, (item: string) => item)
        }
        .key("SwiperF")
        .index(0)
        .indicator(this.indicator)
        .cachedCount(1)
        .loop(false)
        .duration(1000)
        .vertical(false)
        .onChange((index: number) => {
          console.info(index.toString())
        })
      }
    }
    .height('100%')
    .width('80%')
    .margin({ top: 5 })
  }
}