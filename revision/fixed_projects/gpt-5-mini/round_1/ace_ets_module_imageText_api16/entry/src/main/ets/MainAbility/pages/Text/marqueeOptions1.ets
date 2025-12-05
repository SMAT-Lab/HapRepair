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
struct marqueeOptions1 {
  @State mes: string = '';

  build() {
    Row() {
      Column({ space: 8 }) {
        Text(this.mes)
          .id('MarqueeState')
        Text('This is the text with the text overflow set marquee')
          .id('MarqueeOptions')
          .width(300)
          .borderWidth(1)
          .textOverflow({ overflow: TextOverflow.MARQUEE })
          .marqueeOptions({
            start: true,
            fromStart: true,
            step: 6,
            loop: -1,
            delay: 0,
            fadeout: false,
            marqueeStartPolicy: MarqueeStartPolicy.DEFAULT
          })
          .onMarqueeStateChange((state: MarqueeState) => {
            if (state == MarqueeState.START) {
              this.mes = 'START';
            }
          })
      }.width('100%')
    }.height('100%')
  }
}