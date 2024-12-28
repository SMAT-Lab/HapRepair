

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

@Component
struct UserInfo {
  @State private isFollowed: boolean = false

  build() {
    Row() {
      Image($r('app.media.icon_customer'))
        .width(50)
        .height(50)
        .borderRadius(25)
        .backgroundColor('#fcfcfc')
      Text('某某用户').fontSize(16).margin({ left: 15 })
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
    .width('100%')
    .padding(10)
  }
}

@Component
struct MessageContent {
  private imageList: Array<Resource> = [$r('app.media.icon_comment_thumbsup'), $r('app.media.icon_comment_thumbsup')]
  build() {
    Column() {
      Text('今天做的这几道菜，孩子都很喜欢吃。').fontSize(16).margin({ left: 20, right: 20 })
      if (this.imageList.length > 0) {
        Row() {
          Image($r('app.media.dish1')).width(80).height(80).margin(10).objectFit(ImageFit.Cover)
          Image($r('app.media.dish2')).width(80).height(80).margin(10).objectFit(ImageFit.Cover)
          Image($r('app.media.dish3')).width(80).height(80).margin(10).objectFit(ImageFit.Cover)
        }
      }
    }
    .margin(10)
    .alignItems(HorizontalAlign.Start)
  }
}

@Component
struct KeyPanel {
  private isMine: boolean = false
  @State private isThumbsUP: boolean = false
  build() {
    Row({ space: 50 }) {
      Row() {
        Image($r('app.media.icon_comment_thumbsup')).width(25).height(25).fillColor(this.isThumbsUP ? '#cd2f27' : '#000000')
        Text('点赞').fontSize(14).margin({ left: 5 })
      }
      .onClick(() => {
        this.isThumbsUP = !this.isThumbsUP
      })
      Row() {
        Image($r('app.media.icon_comment')).width(25).height(25)
        Text('评论').fontSize(14).margin({ left: 5 })
      }
      if (this.isMine) {
        Row() {
          Image($r('app.media.icon_comment_delete')).width(25).height(25)
          Text('删除').fontSize(14).margin({ left: 5 })
        }
      } else {
        Row() {
          Image($r('app.media.icon_comment_report')).width(25).height(25)
          Text('举报').fontSize(14).margin({ left: 5 })
        }
      }
    }
    .margin({ left: 20, bottom: 10 })
  }
}

@Component
struct MessageItem {
  build() {
    Column() {
      UserInfo()
      MessageContent()
      KeyPanel()
    }
    .alignItems(HorizontalAlign.Start)
    .margin(10)
    .borderRadius(10)
    .backgroundColor('#e6e6e6')
  }
}

@Entry
@Component
export struct MessageList {
  @State private showButton: boolean = false
  build() {
    Column() {
      Text('美食交流圈')
        .width('100%')
        .fontSize(30)
        .textAlign(TextAlign.Center)
        .backgroundColor('#f2f3f5')

      List() {
        ListItem() {
          MessageItem()
        }
        ListItem() {
          MessageItem()
        }
        ListItem() {
          MessageItem()
        }
        ListItem() {
          MessageItem()
        }
        ListItem() {
          MessageItem()
        }
        ListItem() {
          MessageItem()
        }
      }
      .width('100%')
      .alignListItem(ListItemAlign.Center)

      Button({ type: ButtonType.Circle, stateEffect: true }) {
        Image($r('app.media.icon_manage')).width(25).height(25)
      }
      .width(50)
      .height(50)
      .position({ x: '77%', y: '85%' })
      .borderRadius(30)
      .borderWidth(1)
      .backgroundColor('#fcfcfc')
      .onClick(() => {
        this.showButton = !this.showButton
      })

      if (this.showButton) {
        Button({ type: ButtonType.Circle, stateEffect: true }) {
          Text('我发的').fontSize(16)
        }
        .width(60)
        .height(60)
        .position({ x: '54%', y: '86%' })
        .borderRadius(30)
        .borderWidth(1)
        .backgroundColor('#fcfcfc')
        .onClick(() => {
          console.log('筛选自己发布的交流圈')
          this.showButton = !this.showButton
        })

        Button({ type: ButtonType.Circle, stateEffect: true }) {
          Text('我的\n关注').fontSize(14)
        }
        .width(60)
        .height(60)
        .position({ x: '60%', y: '78%' })
        .borderRadius(30)
        .borderWidth(1)
        .backgroundColor('#fcfcfc')
        .onClick(() => {
          console.log('筛选关注的用户的交流圈')
          this.showButton = !this.showButton
        })

        Button({ type: ButtonType.Circle, stateEffect: true }) {
          Text('新建').fontSize(16)
        }
        .width(60)
        .height(60)
        .position({ x: '77%', y: '74%' })
        .borderRadius(30)
        .borderWidth(1)
        .backgroundColor('#fcfcfc')
        .onClick(() => {
          console.log('进入编辑页面')
          this.showButton = !this.showButton
        })
      }
    }
  }
}