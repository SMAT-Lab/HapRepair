



export enum CategoryId {
  Fruit = 0,
  Vegetable,
  Nut,
  Seafood,
  Dessert
}

export type Category = {
  name: Resource
  id: CategoryId
}

export type FoodInfo = {
  id: number
  letter: string
  name: string | Resource
  image: Resource
  categoryId: CategoryId
  calories: number
  protein: number
  fat: number
  carbohydrates: number
  vitaminC: number
}

export enum MealTimeId {
  Breakfast = 0,
  Lunch,
  Dinner,
  Supper
}

export class MealTime {
  name: Resource
  id: MealTimeId

  constructor(id: MealTimeId) {
    this.id = id
    switch (id) {
      case MealTimeId.Breakfast:
        this.name = $r('app.string.meal_time_breakfast')
        break
      case MealTimeId.Lunch:
        this.name = $r('app.string.meal_time_lunch')
        break
      case MealTimeId.Dinner:
        this.name = $r('app.string.meal_time_dinner')
        break
      case MealTimeId.Supper:
        this.name = $r('app.string.meal_time_supper')
        break
    }
  }
}

export class DietRecord {
  id: number
  foodId: number
  mealTime: MealTime
  weight: number

  constructor(id: number, foodId: number, mealTime: MealTime, weight: number) {
    this.id = id
    this.foodId = foodId
    this.mealTime = mealTime
    this.weight = weight
  }
}

export class MealFoodInfo {
  recordId: number
  name: string | Resource
  image: Resource
  calories: number
  protein: number
  fat: number
  carbohydrates: number
  weight: number

  constructor(recordId: number, name: string | Resource, image: Resource, calories: number, protein: number, fat: number, carbohydrates: number, weight: number) {
    this.recordId = recordId
    this.name = name
    this.image = image
    this.calories = calories
    this.protein = protein
    this.fat = fat
    this.carbohydrates = carbohydrates
    this.weight = weight
  }
}

@Observed
export class OneMealStatisticsInfo {
  mealTime: MealTime
  mealFoods: Array<MealFoodInfo> = []
  totalCalories: number = 0
  totalFat: number = 0
  totalCarbohydrates: number = 0
  totalProtein: number = 0

  constructor(mealTime: MealTime) {
    this.mealTime = mealTime
  }
}