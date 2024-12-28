



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
import { TabItem, GoodsItem, getLinkData } from '../model/MarketModels';

@Component
struct GoodsList {
  @State index: number = 0;
  @State requestSuccess: boolean = false;
  private tabArray: Array<TabItem> = [];
  @Provide private listArray: Array<GoodsItem> = [];
  private scroller: Scroller = new Scroller();
  aboutToAppear() {
    setTimeout(() => {
      let linkDataItem = getLinkData();
      this.tabArray = linkDataItem.tabArray;
      this.listArray = linkDataItem.listArray;
      this.requestSuccess = true;
    }, 1200)
  }

  build() {
    Row() {
      if (this.requestSuccess) {
        Scroll() {
          Column() {
            ForEach(this.tabArray.map((item1, index1) => {
              return { index: index1, data: item1 };
            }), item => {
              Text(item.data.typeName)
                .width('100%')
                .height(60)
                .fontColor(0x696969)
                .fontSize(14)
                .textAlign(TextAlign.Center)
                .backgroundColor(this.index === item.index ? 0xffffff : null)
                .onClick(() => {
                  if (this.index !== item.index) {
                    this.index = item.index;
                    this.scroller.scrollToIndex(item.data.position);
                  }
                })
            }, item => '' + item.data)
          }.height('100%')
        }.width(80).height('100%').backgroundColor(0xdddddd).scrollBar(BarState.Off)

        List({ scroller: this.scroller }) {
          ForEach(this.listArray, item => {
            if (item.tag) {
              ListItem() {
                Text(item.goodsName)
                  .width('100%')
                  .height(40)
                  .padding({ left: 10 })
                  .fontSize(20)
                  .fontColor(0x696969)
                  .backgroundColor(0xefefef)
              }.sticky(Sticky.Normal)
            } else {
              ListItem() {
                Row() {
                  Image(item.imageUrl)
                    .width(90)
                    .height(90)
                    .margin(5)
                    .borderRadius(10)
                    .objectFit(ImageFit.Cover)
                  Column() {
                    Text(item.goodsName)
                      .fontSize(16)
                      .fontColor(0x363636)
                    Row() {
                      Text(item.price === 0 ? '限量免费' : '￥' + item.price)
                        .fontSize(14)
                        .fontColor(0xff6600)
                      Counter() {
                        Text('0')
                      }.margin({ left: 30 })
                      .onInc(() => {
                        console.log('加一')
                      })
                      .onDec(() => {
                        console.log('减一')
                      })
                    }
                    .margin({ top: 10 })
                  }
                  .alignItems(HorizontalAlign.Start)
                }
              }
            }
          }, item => '' + item)
        }
        .width('100%')
        .height('100%')
        .layoutWeight(1)
        .onScrollIndex((firstIndex: number) => {
          if (this.index !== this.listArray[firstIndex].index) {
            this.index = this.listArray[firstIndex].index
          }
        })
      } else {
        Text('数据加载中……')
          .width('100%')
          .height('100%')
          .textAlign(TextAlign.Center)
      }
    }
  }
}

@Component
struct MarketInfo {
  build() {
    Row() {
      Column() {
        Row() {
          Image($r('app.media.icon_back'))
            .width(20)
            .height(20)
            .margin({ right: 10 })
            .onClick(() => {
              router.back()
            })
          Text('某某超市').fontSize(20)
          Text('评分：4.2分').fontSize(12).margin({ left: 20 })
        }.margin({ bottom: 20 })
        Text('节日期间，全场商品九折优惠，购物满200元送鸡蛋一盒').fontSize(14).maxLines(2)
      }
      .alignItems(HorizontalAlign.Start)
      .width('65%')
      .margin({ left: 20 })
      Blank()
      Image($r('app.media.logo_shop2'))
        .width(80)
        .height(80)
        .margin(10)
        .borderRadius(10)
    }
    .width('100%')
    .height(80)
    .margin(10)
    .borderRadius(10)
  }
}

@Component
struct CommentItem {
  build() {
    Column() {
      Row() {
        Image($r('app.media.icon_customer'))
          .width(40)
          .height(40)
          .margin(10)
          .borderRadius(20)
          .backgroundColor('#fcfcfc')
        Column() {
          Text('匿名用户').fontSize(16).margin({ bottom: 5 })
          Text('2022-09-24').fontSize(14)
        }.alignItems(HorizontalAlign.Start)
        Blank()
        Image($r('app.media.icon_more'))
          .width(20)
          .height(20)
          .margin({ right: 20 })
      }
      .width('100%')
      .height(60)
      Row() {
        Rating({ rating: 4, indicator: true })
          .stars(5)
          .stepSize(0.5)
          .width(75)
          .height(15)
        Text('满意度: 比较满意').fontSize(12).margin({ left: 20 })
      }
      .height(20)
      .margin({ left: 20 })
      Text('经常在这家买东西，水果蔬菜都挺新鲜，只是配送速度有些慢。').fontSize(16).margin(10)
    }
    .alignItems(HorizontalAlign.Start)
    .width('100%')
    .borderRadius(10)
    .backgroundColor('#e6e6e6')
  }
}

@Component
struct Comment {
  build() {
    List({ space: 10, initialIndex: 0 }) {
      ListItem() {
        CommentItem()
      }
      ListItem() {
        CommentItem()
      }
      ListItem() {
        CommentItem()
      }
      ListItem() {
        CommentItem()
      }
      ListItem() {
        CommentItem()
      }
      ListItem() {
        CommentItem()
      }
    }
    .width('100%')
    .height('100%')
    .padding(15)
    .alignListItem(ListItemAlign.Center)
  }
}

@Component
struct StoreFront {
  build() {
    Column({ space: 20 }) {
      Row() {
        Column() {
          Text('某某超市').fontSize(20).margin({ left: 10, top: 10 })
          Row() {
            Image($r('app.media.icon_location')).width(20).height(20).margin(10)
            Text('广东省惠州市惠城区某某镇某某路5号').fontSize(14).fontColor('#696969')
          }
        }
        .alignItems(HorizontalAlign.Start)
        Blank()
        Image($r('app.media.icon_call')).width(40).height(40).margin(10)
      }
      .width('100%').margin({ top: 20 })

      Column() {
        Text('店面环境').fontSize(20).margin({ left: 10, top: 10 })
        Row() {
          Image($r('app.media.logo_shop1'))
            .width(80)
            .height(80)
            .margin(10)
            .borderRadius(10)
          Image($r('app.media.logo_shop2'))
            .width(80)
            .height(80)
            .margin(10)
            .borderRadius(10)
          Image($r('app.media.logo_shop3'))
            .width(80)
            .height(80)
            .margin(10)
            .borderRadius(10)
        }
      }
      .width('100%').alignItems(HorizontalAlign.Start)
      Column({ space: 20 }) {
        Text('商家信息').fontSize(20).margin({ left: 10, top: 10 })
        Text('商家品类：生鲜超市').fontSize(16).fontColor('#696969').margin({ left: 10 })
        Text('营业时间：周一至周日06:00-22:00').fontSize(16).fontColor('#696969').margin({ left: 10 })
      }
      .width('100%').alignItems(HorizontalAlign.Start)
      Button('查看本店更多信息', { type: ButtonType.Capsule, stateEffect: false }).opacity(0.5)
        .backgroundColor('#317aff').width(200)
    }
  }
}

@Component
struct ShopCart {
  build() {
    Row() {
      Image($r('app.media.icon_cart'))
        .width(40)
        .height(40)
        .objectFit(ImageFit.Contain)
        .margin({ left: 10, right: 10 })
      Text('￥0')
        .fontSize(20)
      Blank()
      Button() {
        Text('提交订单').fontSize(16)
      }.width(90).height(30).margin({ right: 10 }).backgroundColor('#4395ff')
    }
    .width('100%')
    .backgroundColor('#fcfcfc')
  }
}

@Entry
@Component
struct Main {
  build() {
    Column() {
      MarketInfo().height('10%')
      Tabs() {
        TabContent() {
          Column() {
            GoodsList().height('92%')
            ShopCart().height('8%')
          }
        }.tabBar('全部商品').backgroundColor('#fcfcfc')
        TabContent() {
          Comment()
        }.tabBar('评价').backgroundColor('#fcfcfc')
        TabContent() {
          StoreFront()
        }.tabBar('商家').backgroundColor('#fcfcfc')
      }
      .height('90%')
      .scrollable(false)
      .barWidth(280)
      .barMode(BarMode.Scrollable)
    }
    .backgroundColor('#b4f1fe')
  }
}