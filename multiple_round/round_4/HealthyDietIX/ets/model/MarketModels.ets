



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

const GoodsData: any[] = [
  { 'typeId': 1, 'typeName': '时令蔬菜', 'id': 1, 'goodsName': '白菜', 'imageUrl': $r('app.media.tomato'), 'price': 1.8 },
  { 'typeId': 1, 'typeName': '时令蔬菜', 'id': 2, 'goodsName': '胡萝卜', 'imageUrl': $r('app.media.tomato'), 'price': 1.8 },
  { 'typeId': 1, 'typeName': '时令蔬菜', 'id': 3, 'goodsName': '生菜', 'imageUrl': $r('app.media.tomato'), 'price': 1.8 },
  { 'typeId': 1, 'typeName': '时令蔬菜', 'id': 4, 'goodsName': '芹菜', 'imageUrl': $r('app.media.tomato'), 'price': 1.8 },
  { 'typeId': 1, 'typeName': '时令蔬菜', 'id': 5, 'goodsName': '上海青', 'imageUrl': $r('app.media.tomato'), 'price': 1.8 },
  { 'typeId': 1, 'typeName': '时令蔬菜', 'id': 6, 'goodsName': '娃娃菜', 'imageUrl': $r('app.media.tomato'), 'price': 1.8 },
  { 'typeId': 1, 'typeName': '时令蔬菜', 'id': 7, 'goodsName': '苦瓜', 'imageUrl': $r('app.media.tomato'), 'price': 1.8 },
  { 'typeId': 1, 'typeName': '时令蔬菜', 'id': 8, 'goodsName': '冬瓜', 'imageUrl': $r('app.media.tomato'), 'price': 1.8 },
  { 'typeId': 1, 'typeName': '时令蔬菜', 'id': 9, 'goodsName': '黄瓜', 'imageUrl': $r('app.media.tomato'), 'price': 1.8 },
  { 'typeId': 2, 'typeName': '肉禽鱼蛋', 'id': 10, 'goodsName': '猪肉', 'imageUrl': $r('app.media.tomato'), 'price': 1.8 },
  { 'typeId': 2, 'typeName': '肉禽鱼蛋', 'id': 11, 'goodsName': '鸡肉', 'imageUrl': $r('app.media.tomato'), 'price': 1.8 },
  { 'typeId': 2, 'typeName': '肉禽鱼蛋', 'id': 12, 'goodsName': '草鱼', 'imageUrl': $r('app.media.tomato'), 'price': 1.8 },
  { 'typeId': 2, 'typeName': '肉禽鱼蛋', 'id': 13, 'goodsName': '土鸡蛋', 'imageUrl': $r('app.media.tomato'), 'price': 1.8 },
  { 'typeId': 3, 'typeName': '新鲜水果', 'id': 14, 'goodsName': '苹果', 'imageUrl': $r('app.media.tomato'), 'price': 1.8 },
  { 'typeId': 3, 'typeName': '新鲜水果', 'id': 15, 'goodsName': '香蕉', 'imageUrl': $r('app.media.tomato'), 'price': 1.8 },
  { 'typeId': 3, 'typeName': '新鲜水果', 'id': 16, 'goodsName': '梨', 'imageUrl': $r('app.media.tomato'), 'price': 1.8 },
  { 'typeId': 3, 'typeName': '新鲜水果', 'id': 17, 'goodsName': '葡萄', 'imageUrl': $r('app.media.tomato'), 'price': 1.8 },
  { 'typeId': 3, 'typeName': '新鲜水果', 'id': 18, 'goodsName': '西瓜', 'imageUrl': $r('app.media.tomato'), 'price': 1.8 },
  { 'typeId': 4, 'typeName': '酒水饮料', 'id': 19, 'goodsName': '可乐', 'imageUrl': $r('app.media.tomato'), 'price': 1.8 },
  { 'typeId': 4, 'typeName': '酒水饮料', 'id': 20, 'goodsName': '橙汁', 'imageUrl': $r('app.media.tomato'), 'price': 1.8 },
  { 'typeId': 4, 'typeName': '酒水饮料', 'id': 21, 'goodsName': '菠萝啤', 'imageUrl': $r('app.media.tomato'), 'price': 1.8 },
  { 'typeId': 5, 'typeName': '休闲零食', 'id': 22, 'goodsName': '辣条', 'imageUrl': $r('app.media.tomato'), 'price': 1.8 },
  { 'typeId': 5, 'typeName': '休闲零食', 'id': 23, 'goodsName': '薯片', 'imageUrl': $r('app.media.tomato'), 'price': 1.8 },
  { 'typeId': 5, 'typeName': '休闲零食', 'id': 24, 'goodsName': '棒棒糖', 'imageUrl': $r('app.media.tomato'), 'price': 1.8 },
  { 'typeId': 5, 'typeName': '休闲零食', 'id': 25, 'goodsName': '口香糖', 'imageUrl': $r('app.media.tomato'), 'price': 1.8 },
  { 'typeId': 5, 'typeName': '休闲零食', 'id': 26, 'goodsName': '饼干', 'imageUrl': $r('app.media.tomato'), 'price': 1.8 },
  { 'typeId': 5, 'typeName': '休闲零食', 'id': 27, 'goodsName': '巧克力', 'imageUrl': $r('app.media.tomato'), 'price': 1.8 },
  { 'typeId': 5, 'typeName': '休闲零食', 'id': 28, 'goodsName': '瓜子', 'imageUrl': $r('app.media.tomato'), 'price': 1.8 },
  { 'typeId': 6, 'typeName': '调味料', 'id': 29, 'goodsName': '生抽', 'imageUrl': $r('app.media.tomato'), 'price': 1.8 },
  { 'typeId': 6, 'typeName': '调味料', 'id': 30, 'goodsName': '陈醋', 'imageUrl': $r('app.media.tomato'), 'price': 1.8 },
  { 'typeId': 6, 'typeName': '调味料', 'id': 31, 'goodsName': '鸡精', 'imageUrl': $r('app.media.tomato'), 'price': 1.8 },
  { 'typeId': 6, 'typeName': '调味料', 'id': 32, 'goodsName': '胡椒粉', 'imageUrl': $r('app.media.tomato'), 'price': 1.8 },
  { 'typeId': 6, 'typeName': '调味料', 'id': 33, 'goodsName': '豆瓣酱', 'imageUrl': $r('app.media.tomato'), 'price': 1.8 },
  { 'typeId': 6, 'typeName': '调味料', 'id': 34, 'goodsName': '蒜蓉酱', 'imageUrl': $r('app.media.tomato'), 'price': 1.8 },
  { 'typeId': 6, 'typeName': '调味料', 'id': 35, 'goodsName': '辣椒酱', 'imageUrl': $r('app.media.tomato'), 'price': 1.8 },
  { 'typeId': 7, 'typeName': '日用品', 'id': 36, 'goodsName': '纸巾', 'imageUrl': $r('app.media.tomato'), 'price': 1.8 },
  { 'typeId': 7, 'typeName': '日用品', 'id': 37, 'goodsName': '洗洁精', 'imageUrl': $r('app.media.tomato'), 'price': 1.8 },
  { 'typeId': 7, 'typeName': '日用品', 'id': 38, 'goodsName': '洗衣液', 'imageUrl': $r('app.media.tomato'), 'price': 1.8 },
  { 'typeId': 7, 'typeName': '日用品', 'id': 39, 'goodsName': '洗发精', 'imageUrl': $r('app.media.tomato'), 'price': 1.8 },
  { 'typeId': 7, 'typeName': '日用品', 'id': 40, 'goodsName': '沐浴露', 'imageUrl': $r('app.media.tomato'), 'price': 1.8 },
  { 'typeId': 7, 'typeName': '日用品', 'id': 41, 'goodsName': '牙刷', 'imageUrl': $r('app.media.tomato'), 'price': 1.8 },
  { 'typeId': 7, 'typeName': '日用品', 'id': 42, 'goodsName': '毛巾', 'imageUrl': $r('app.media.tomato'), 'price': 1.8 },
  { 'typeId': 7, 'typeName': '日用品', 'id': 43, 'goodsName': '牙膏', 'imageUrl': $r('app.media.tomato'), 'price': 1.8 },
  { 'typeId': 7, 'typeName': '日用品', 'id': 44, 'goodsName': '一次性筷子', 'imageUrl': $r('app.media.tomato'), 'price': 1.8 },
  //  { 'typeId': 1, 'typeName': '商品类别', 'id': 1, 'goodsName': '商品名', 'imageUrl': $r('app.media.tomato'), 'price': 1.8 },
]

export class TabItem {
  position: number;
  typeName: string;
  constructor(position: number, typeName: string) {
    this.position = position;
    this.typeName = typeName;
  }
}

export class GoodsItem {
  tag: boolean;
  index: number;
  goodsName: string;
  imageUrl: string;
  price: number;
  count: number;
  constructor(tag: boolean, index: number, goodsName: string, imageUrl: string, price: number, count: number) {
    this.tag = tag;
    this.index = index;
    this.goodsName = goodsName;
    this.imageUrl = imageUrl;
    this.price = price;
    this.count = count;
  }
}

export class LinkDataItem {
  tabArray: Array<TabItem> = []
  listArray: Array<GoodsItem> = []
  constructor(tabArray: Array<TabItem>, listArray: Array<GoodsItem>) {
    this.tabArray = tabArray;
    this.listArray = listArray;
  }
}

export function getLinkData(): LinkDataItem {
  let linkDataItem = new LinkDataItem([], []);
  let typeId: number = 0;
  GoodsData.forEach(item => {
    if (typeId !== item.typeId) {
      let tabItem = new TabItem(linkDataItem.listArray.length, item.typeName);
      linkDataItem.tabArray.push(tabItem)
      let goodsItem = new GoodsItem(true, linkDataItem.tabArray.length - 1, item.typeName, '', 0, 0);
      linkDataItem.listArray.push(goodsItem)
    }
    let goodsItem = new GoodsItem(false, linkDataItem.tabArray.length - 1, item.goodsName, item.imageUrl, item.price, item.count);
    linkDataItem.listArray.push(goodsItem);
    typeId = item.typeId;
  })
  return linkDataItem;
}