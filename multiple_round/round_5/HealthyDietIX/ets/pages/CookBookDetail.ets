



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

@Component
struct PageTitle {
  private dishName: string = '水煮鸡胸'
  @State private isCollected: boolean = false

  build() {
    Row() {
      Image($r('app.media.back'))
        .width(20)
        .height(20)
        .onClick(() => {
          router.back()
        })
      Text(this.dishName)
        .fontSize(22)
        .margin({ left: 20 })
      Blank()
      Button(this.isCollected ? '已收藏' : '收藏', { type: ButtonType.Capsule, stateEffect: true })
        .width(80)
        .height(40)
        .margin({ right: 10 })
        .borderRadius(8)
        .backgroundColor(this.isCollected ? '#74cd57' : '#a2a2a2')
        .onClick(() => {
          this.isCollected = !this.isCollected
        })
    }
    .padding(12)
    .width('100%')
  }
}

@Component
struct UserInfo {
  private userIcon: Resource = $r('app.media.icon_customer')
  private userName: string = '六星厨师'
  @State private isFollowed: boolean = false

  build() {
    Row() {
      Image(this.userIcon)
        .width(50)
        .height(50)
      Text(this.userName)
        .fontSize(18)
        .margin({ left: 20 })
      Blank()
      Button(this.isFollowed ? '已关注' : '关注', { type: ButtonType.Capsule, stateEffect: true })
        .width(80)
        .height(40)
        .margin({ right: 10 })
        .borderRadius(8)
        .backgroundColor(this.isFollowed ? '#74cd57' : '#a2a2a2')
        .onClick(() => {
          this.isFollowed = !this.isFollowed
        })
    }
    .padding(12)
    .width('100%')
  }
}

@Component
struct SingleStepper {
  private index: number
  private operation: string
  build() {
    Column() {
      Text(`第${this.index}步`).fontSize(20).margin(10)
      Text(this.operation).fontSize(16).margin({ left: 20, right: 20 })
    }
    .alignItems(HorizontalAlign.Start)
  }
}

@Entry
@Component
struct CookingGuide {
  private allSteps: Array<string> = [
    '将鸡胸洗净，用温水淹没鸡肉，放入适量白胡椒粉和百里香腌制30~60分钟',
    '将鸡肉联通腌制的水一起倒入锅中，开大火盖上锅盖',
    '煮五分钟，期间偶尔翻动肉块',
    '五分钟后关火，盖好锅盖再闷五分钟',
    '将鸡肉捞出盛到碗里，淋上酱油就大功告成',
  ]

  build() {
    Column() {
      Stack() {
        Image($r('app.media.dish4'))
        PageTitle()
      }
      .alignContent(Alignment.TopStart)
      .width('100%')
      .height(200)
      UserInfo()
      List() {
        ForEach(this.allSteps, (item, index) => {
          ListItem() {
            SingleStepper({ index: index+1, operation: item })
          }
        }, (item, index) => index)
      }
     .height(100)
     .width(100)
    }
  }
}