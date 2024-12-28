

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
import { BreakPointType } from '../common/BreakpointSystem';

@Component
struct DishItem {
  private dishItem: DishInfo

  build() {
    Column() {
      Image(this.dishItem.image)
        .objectFit(ImageFit.Cover)
        .backgroundColor('#f1f3f5')
        .height(130)
      Text(this.dishItem.name)
        .fontSize(16)
    }
    .width(150)
    .height(150)
    .margin(10)
    .clip(new Rect({ width: '100%', height: '100%', radius: 10 }))
    .backgroundColor('#e8e8e8')
    .onClick(() => {
      router.push({ url: 'pages/CookBookDetail', params: { dishId: this.dishItem } })
    })
  }
}

@Component
struct DishesGrid {
  @StorageProp('currentBreakpoint') currentBreakpoint: string = 'md'
  private dishItems: DishInfo[]

  build() {
    Grid() {
      ForEach(this.dishItems, (item: DishInfo) => {
        GridItem() {
          DishItem({ dishItem: item })
        }
      }, (item: DishInfo) => item.id)
    }
    .columnsTemplate(new BreakPointType({
      sm: '1fr 1fr',
      md: '1fr 1fr 1fr',
      lg: '1fr 1fr 1fr 1fr'
    }).getValue(this.currentBreakpoint))
    .columnsGap(8)
    .rowsGap(8)
    .padding({ left: 16, right: 16 })
  }
}

export enum DishesCategoryId {
  HomeCooking = 0,
  Soup,
  Cakes,
  FitnessMeal
}

export type DishesCategory = {
  id: DishesCategoryId
  name: string
}

export type DishInfo = {
  id: number
  name: string
  image: Resource
  categoryId: DishesCategoryId
}

@Entry
@Component
export struct CookBook {
  @State currentTabIndex: number = 0
  private dishItems: Array<DishInfo> = [
    { id: 0, name: '酸辣土豆丝', image: $r('app.media.dish1'), categoryId: DishesCategoryId.HomeCooking },
    { id: 1, name: '玉米排骨汤', image: $r('app.media.dish2'), categoryId: DishesCategoryId.Soup },
    { id: 2, name: '麻薯', image: $r('app.media.dish3'), categoryId: DishesCategoryId.Cakes },
    { id: 3, name: '水煮鸡胸', image: $r('app.media.dish4'), categoryId: DishesCategoryId.FitnessMeal },
  ]
  private foodCategories: Array<DishesCategory> = [
    { id: 0, name: '家常菜' },
    { id: 1, name: '炖汤' },
    { id: 2, name: '糕点' },
    { id: 3, name: '健身餐' },
  ]

  @Builder tabBarItemBuilder(value: string, index: number) {
    Text(value)
      .fontColor(this.currentTabIndex === index ? 'rgba(0,0,0,0.9)' : 'rgba(0,0,0,0.6)')
      .fontSize(this.currentTabIndex === index ? 24 : 18)
      .margin({ top: 2 })
      .height(56)
  }

  build() {
    Stack() {
      Tabs() {
        TabContent() {
          DishesGrid({ dishItems: this.dishItems })
        }.tabBar(this.tabBarItemBuilder('全部', 0))
        ForEach(this.foodCategories, (dishCategory: DishesCategory, index) => {
          TabContent() {
            DishesGrid({ dishItems: this.dishItems.filter(item => (item.categoryId === dishCategory.id)) })
          }.tabBar(this.tabBarItemBuilder(dishCategory.name, index + 1))
        }, (dishCategory: DishesCategory, index) => dishCategory.id)
      }
      .animationDuration(0)
      .barMode(BarMode.Scrollable)
      .onChange((index) => {
        this.currentTabIndex = index
      })

      Button({ type: ButtonType.Circle, stateEffect: true }) {
        Image($r('app.media.icon_newone')).width(25).height(25)
      }
      .width(50)
      .height(50)
      .position({ x: '75%', y: '85%' })
      .borderRadius(30)
      .borderWidth(1)
      .backgroundColor('#fcfcfc')
      .onClick(() => {
        router.push({ url: 'pages/CookBookCreate' })
      })
    }
  }
}