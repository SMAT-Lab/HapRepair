


import router from '@ohos.router';
import { FoodInfo, Category, CategoryId, MealTimeId, OneMealStatisticsInfo, MealFoodInfo, DietRecord } from './DataModels';
import { mockFoods, mockFoodInfo, mockDietRecords } from '../mock/MockData';

const DEBUG_PREVIEW = false
const MOCK_API = true

export function getFoods(): Array<FoodInfo> {
  return mockFoods
}

export function getFoodInfo(keyName: string): FoodInfo {
  return DEBUG_PREVIEW ? mockFoodInfo : router.getParams()[keyName]
}

export let initDietRecords: Array<DietRecord> = [
  {
    id: -1,
    foodId: 0,
    mealTime: { name: $r('app.string.meal_time_breakfast'), id: MealTimeId.Breakfast },
    weight: 0
  },
  {
    id: -1,
    foodId: 0,
    mealTime: { name: $r('app.string.meal_time_lunch'), id: MealTimeId.Lunch },
    weight: 0
  },
  {
    id: -1,
    foodId: 0,
    mealTime: { name: $r('app.string.meal_time_dinner'), id: MealTimeId.Dinner },
    weight: 0
  },
  {
    id: -1,
    foodId: 0,
    mealTime: { name: $r('app.string.meal_time_supper'), id: MealTimeId.Supper },
    weight: 0
  }
]

export function getDietRecords(): Array<DietRecord> {
  return DEBUG_PREVIEW ? initDietRecords.concat(mockDietRecords) : AppStorage.Get<Array<DietRecord>>('dietRecords')
}

export function getFoodCategories(): Category[] {
  return [
    { id: CategoryId.Vegetable, name: $r('app.string.category_vegetable') },
    { id: CategoryId.Fruit, name: $r('app.string.category_fruit') },
    { id: CategoryId.Nut, name: $r('app.string.category_nut') },
    { id: CategoryId.Seafood, name: $r('app.string.category_seafood') },
    { id: CategoryId.Dessert, name: $r('app.string.category_dessert') }
  ]
}

export function getMileTimes(): string[] {
  if (MOCK_API) {
    return ['早餐', '午餐', '晚餐', '夜宵']
  } else {
    let mealTimeResources: Resource[] = [$r("app.string.meal_time_breakfast"), $r('app.string.meal_time_lunch'), $r('app.string.meal_time_dinner'), $r('app.string.meal_time_supper'), $r('app.string.category_dessert')]
    let mealTimes = []
    mealTimeResources.forEach(item => {
      let mealTime = this.context.resourceManger.getStringSync(item.id)
      if (mealTime !== '') {
        mealTimes.push(mealTime)
      }
    })
    return mealTimes
  }
}

export function getSortedFoodData(): Array<FoodInfo> {
  let letterReg = /^[A-Z]$/
  let list = new Array()
  let foods = getFoods()
  for (let i=0; i<foods.length; i++) {
    list['#'] = new Array()
    let letter = foods[i].letter.substr(0, 1).toUpperCase()
    if (!letterReg.test(letter)) {
      letter = '#'
    }
    if (!(letter in list)) {
      list[letter] = new Array()
    }
    list[letter].push(foods[i])
  }

  let letterGroup = new Array()
  for (let key in list) {
    letterGroup.push({ letter: key, list: list[key] })
  }
  letterGroup.sort(function(x, y) {
    return x.letter.charCodeAt(0) - y.letter.charCodeAt(0)
  })

  let lastArr = letterGroup[0]
  letterGroup.splice(0, 1)
  letterGroup.push(lastArr)

  let resultList = []
  for (let i=0; i<letterGroup.length; i++) {
    resultList.push(letterGroup[i].letter)
    resultList = resultList.concat(letterGroup[i].list)
  }
  return resultList
}

export function statistics(): Array<OneMealStatisticsInfo> {
  console.info('meal statistics')
  let dietRecords = getDietRecords()
  const mealMap = new Map()
  dietRecords.forEach((item) => {
    let oneMealStatisticsInfo: OneMealStatisticsInfo = mealMap.get(item.mealTime.id)
    if (oneMealStatisticsInfo === undefined) {
      oneMealStatisticsInfo = new OneMealStatisticsInfo(item.mealTime)
    }
    let foodInfo = getFoods().find((food) => {
      return food.id === item.foodId
    })
    let calories = foodInfo.calories * item.weight
    let protein = foodInfo.protein * item.weight
    let fat = foodInfo.fat * item.weight
    let carbohydrates = foodInfo.carbohydrates * item.weight
    oneMealStatisticsInfo.mealFoods.push(new MealFoodInfo(item.id, foodInfo.name, foodInfo.image, calories, protein, fat, carbohydrates, item.weight))
    oneMealStatisticsInfo.totalFat += fat
    oneMealStatisticsInfo.totalCalories += calories
    oneMealStatisticsInfo.totalCarbohydrates += carbohydrates
    oneMealStatisticsInfo.totalProtein += protein
    mealMap.set(item.mealTime.id, oneMealStatisticsInfo)
  })
  return Array.from(mealMap.values())
}

export function updateDietWeight(recordId: number, weight: number) {
  let dietRecords = getDietRecords()
  let index = dietRecords.findIndex((record) => {
    return record.id === recordId
  })
  dietRecords[index].weight = weight
  AppStorage.SetOrCreate<Array<DietRecord>>('dietRecords', dietRecords)
}



export function getCartRecords(): Array<DietRecord> {
  return DEBUG_PREVIEW ? initDietRecords.concat(mockDietRecords) : AppStorage.Get<Array<DietRecord>>('dietRecords')
}

export function amount(): Array<OneMealStatisticsInfo> {
  console.info('calculate the total price')
  let dietRecords = getCartRecords()
  const mealMap = new Map()
  dietRecords.forEach((item) => {
    let oneMealStatisticsInfo: OneMealStatisticsInfo = mealMap.get(item.mealTime.id)
    if (oneMealStatisticsInfo === undefined) {
      oneMealStatisticsInfo = new OneMealStatisticsInfo(item.mealTime)
    }
    let foodInfo = getFoods().find((food) => {
      return food.id === item.foodId
    })
    let calories = foodInfo.calories * item.weight
    let protein = foodInfo.protein * item.weight
    let fat = foodInfo.fat * item.weight
    let carbohydrates = foodInfo.carbohydrates * item.weight
    oneMealStatisticsInfo.mealFoods.push(new MealFoodInfo(item.id, foodInfo.name, foodInfo.image, calories, protein, fat, carbohydrates, item.weight))
    oneMealStatisticsInfo.totalFat += fat
    oneMealStatisticsInfo.totalCalories += calories
    oneMealStatisticsInfo.totalCarbohydrates += carbohydrates
    oneMealStatisticsInfo.totalProtein += protein
    mealMap.set(item.mealTime.id, oneMealStatisticsInfo)
  })
  return Array.from(mealMap.values())
}

export function updateGoodCount(recordId: number, weight: number) {
  let dietRecords = getDietRecords()
  let index = dietRecords.findIndex((record) => {
    return record.id === recordId
  })
  dietRecords[index].weight = weight
  AppStorage.SetOrCreate<Array<DietRecord>>('dietRecords', dietRecords)
}