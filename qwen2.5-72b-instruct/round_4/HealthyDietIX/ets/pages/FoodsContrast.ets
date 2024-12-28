



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
import { FoodInfo, CategoryId, MealTime, MealTimeId, DietRecord } from '../model/DataModels'
import { getFoodInfo, initDietRecords, getMileTimes , getSortedFoodData} from '../model/DataUtil'
import { mockFoods } from '../mock/MockData';

@Component
struct PageTitle {
  private titleName: Resource = $r('app.string.title_food_contrast')

  build() {
    Row() {
      Image($r('app.media.back'))
        .width(20)
        .height(20)
        .onClick(() => {
          router.back()
        })
      Text(this.titleName)
        .fontSize(22)
        .margin({ left: 20 })
    }
    .padding(12)
    .width('100%')
  }
}

@Component
struct OneVsOne {
  private imageOne: Resource
  private imageTwo: Resource
  build() {
    Row() {
      Image(this.imageOne).width(120).height(120)
      Text('VS').fontSize(20).margin({ left: 30, right: 30 })
      Image(this.imageTwo).width(120).height(120)
    }
  }
}

@Component
struct CompareItem {
  private ValueOne: number = 0
  private ValueTwo: number = 0
  private ElementItem: Resource
  build() {
    Row() {
      Text(this.ValueOne.toString()).layoutWeight(1).fontSize(20).fontColor(this.ValueOne > this.ValueTwo ? '#fe931f' : '#000000')
      Text(this.ElementItem).fontSize(20).layoutWeight(1)
      Text(this.ValueTwo.toString()).layoutWeight(1).fontSize(20).fontColor(this.ValueTwo > this.ValueOne ? '#fe931f' : '#000000')
    }
    .margin(20)
  }
}

@Component
struct ContentCompare {
  private FoodOne: FoodInfo
  private FoodTwo: FoodInfo

  build() {
    Column() {
      Row() {
        Text(this.FoodOne.name).layoutWeight(1).fontSize(24)
        Text('营养元素').layoutWeight(1).fontSize(24)
        Text(this.FoodTwo.name).layoutWeight(1).fontSize(24)
      }.margin({ top: 20, bottom: 30 })
      CompareItem({ ValueOne: this.FoodOne.calories, ElementItem: $r('app.string.diet_record_energy'), ValueTwo: this.FoodTwo.calories })
      CompareItem({ ValueOne: this.FoodOne.protein, ElementItem: $r('app.string.diet_record_protein'), ValueTwo: this.FoodTwo.protein })
      CompareItem({ ValueOne: this.FoodOne.fat, ElementItem: $r('app.string.diet_record_fat'), ValueTwo: this.FoodTwo.fat })
      CompareItem({ ValueOne: this.FoodOne.carbohydrates, ElementItem: $r('app.string.diet_record_carbohydrates'), ValueTwo: this.FoodTwo.carbohydrates })
      CompareItem({ ValueOne: this.FoodOne.vitaminC, ElementItem: $r('app.string.diet_record_vitaminC'), ValueTwo: this.FoodTwo.vitaminC })
    }
  }
}

@Entry
@Component
struct Contrast {
  private FoodOne: FoodInfo = getFoodInfo('FoodOne')
  private FoodTwo: FoodInfo = getFoodInfo('FoodTwo')

  build() {

    Column() {
      PageTitle()
      OneVsOne({ imageOne: this.FoodOne.image, imageTwo: this.FoodTwo.image })
      ContentCompare({ FoodOne: this.FoodOne, FoodTwo: this.FoodTwo })
    }
    
  }
}







