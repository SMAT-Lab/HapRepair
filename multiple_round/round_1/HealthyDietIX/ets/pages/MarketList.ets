/*
 * Copyright (c) 2022 LookerSong
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

import router from '@ohos.router';
export type MarketInfo = {
  logo: string | Resource
  name: string | Resource,
  notice: string,
  star: number
}

export let mockMarkets: Array<MarketInfo> = [
  { logo: $r('app.media.logo_shop1'), name: '某某农贸市场', notice: '品类齐全，欢迎各位前来选购', star: 3.8 },
  { logo: $r('app.media.logo_shop2'), name: '某某超市', notice: '节日期间，全场八折优惠', star: 4.7 },
  { logo: $r('app.media.logo_shop3'), name: '某记士多店', notice: '新店开张，请多关照！', star: 3.5 },
  { logo: $r('app.media.logo_shop1'), name: '某某农贸市场', notice: '品类齐全，欢迎各位前来选购', star: 3.8 },
  { logo: $r('app.media.logo_shop2'), name: '某某超市', notice: '节日期间，全场八折优惠', star: 4.7 },
  { logo: $r('app.media.logo_shop3'), name: '某记士多店', notice: '新店开张，请多关照！', star: 3.5 },
  { logo: $r('app.media.logo_shop1'), name: '某某农贸市场', notice: '品类齐全，欢迎各位前来选购', star: 3.8 },
  { logo: $r('app.media.logo_shop2'), name: '某某超市', notice: '节日期间，全场八折优惠', star: 4.7 },
  { logo: $r('app.media.logo_shop3'), name: '某记士多店', notice: '新店开张，请多关照！', star: 3.5 },
]

@Component
struct MarketItem {
  private Market: MarketInfo
  build() {
    Row() {
      Image(this.Market.logo)
        .width(90)
        .height(90)
        .borderRadius(10)
        .margin({ left: 15, top: 10, bottom: 10 })
      Column() {
        Text(this.Market.name).fontSize(20)
        Text(this.Market.notice).fontSize(14).margin({ top: 5, bottom: 10 })
        Rating({ rating: this.Market.star, indicator: true })
          .stars(5)
          .stepSize(0.5)
          .width(100)
          .height(20)
      }
      .margin({ left: 30 })
      .alignItems(HorizontalAlign.Start)
    }
    .width('100%')
    .onClick(() => {
      router.push({
        url: 'pages/MarketDetail',
//        params: { foodId: this.foodItem }
      })
    })
  }
}

@Entry
@Component
export struct MarketList {
  private AllMarkets: Array<MarketInfo> = mockMarkets

  build() {
    Column() {
      Text('附近的商家')
        .width('100%')
        .fontSize(30)
        .textAlign(TextAlign.Center)
        .backgroundColor('#f2f3f5')
      Scroll() {
        List() {
          ForEach(this.AllMarkets, (item) => {
            ListItem() {
              MarketItem({ Market: item })
            }
          }, item => item)
        }
      }.scrollBar(BarState.Auto)
    }
  }
}