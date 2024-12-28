```arkts

/*
 * Copyright (c) 2022 Huawei Device Co., Ltd.
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
import {GetColor} from "../../../common/components/getColor"
import {NavigationBar} from "../../../common/components/navigationBar"

async function routePageShoppingCar() {
  let options = {
    url: 'pages/commonProperties/polymorphicStyle/shoppingCar'
  }
  try {
    await router.push(options)
  } catch (err) {
    console.error(`fail callback, code: ${err.code}, msg: ${err.msg}`)
  }
}

@Entry
@Component
struct PolymorphicStyleSample {
  isEnable: boolean = true

  @State styleHeight: number = 65
  @State count1: number = 0
  @State count2: number = 0
  @State count3: number = 0
  setSliderStep: number = 0.1;
  initVal: number = 0.5;
  @State opacityVal: number = 10;
  num1: number = 30
  @State itemList: any[] = [
    {
      imgSrc: '/common/mate40.png',
      detail: '麒麟',
      price: '￥ 6199.00',

      shoppingCar: '/common/ic_shopping_cart+.png',
      stock: 10
    },
    {
      imgSrc: '/common/mate40.png',
      detail: '麒麟',
      price: '￥ 6199.00',

      shoppingCar: '/common/ic_shopping_cart+.png',
      stock: 4
    },
    {
      imgSrc: '/common/mate40.png',
      detail: '麒麟',
      price: '￥ 6199.00',

      shoppingCar: '/common/ic_shopping_cart+.png',
      stock: 0
    },
  ]