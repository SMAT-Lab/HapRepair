


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

import prompt from '@ohos.prompt';
import router from '@ohos.router';

@Component
struct PageTitle {
  private titleName: string = '新建菜谱'

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
      Blank()
      Button('发布', { type: ButtonType.Normal, stateEffect: true })
        .width(80)
        .height(40)
        .margin({ right: 10 })
        .borderRadius(8)
        .backgroundColor('#a2a2a2')
        .onClick(() => {
          prompt.showToast({
            message: '功能暂未开发，发布失败……',
            duration: 2000,
          });
        })
    }
    .padding(12)
    .width('100%')
  }
}

@Component
struct DishOutline {
  build() {
    Row() {
      Image($r('app.media.icon_upload'))
        .width(140)
        .height(140)
        .borderWidth(2)
        .margin(20)
        .padding(50)
        .backgroundColor('#ffffff')
        .onClick(() => {
          console.log('上传图片')
        })
      Column({ space: 20 }) {
        Row() {
          Text('分类').fontSize(20).margin({ left: 10 })
          Select([{ value:'家常菜' }, { value:'炖汤' }, { value:'糕点' }, { value:'健身餐' }])
            .width(140)
            .selected(0)
            .value('家常菜')
            .font({size: 20, weight:400, family: 'serif', style: FontStyle.Normal })
            .selectedOptionFont({size: 16, weight: 400, family: 'serif', style: FontStyle.Normal })
            .optionFont({size: 16, weight: 400, family: 'serif', style: FontStyle.Normal })
            .onSelect((index:number)=>{
              console.info("Select:" + index)
            })
        }
        TextInput({ placeholder: '菜品名称' }).width(160)
      }
      .width('50%')
    }
    .width('100%')
  }
}

@Component
struct CookingStep {
  build() {
    Column() {
      Row() {
        Text('步骤n').fontSize(20).margin({ left: 20 })
        Blank()
        Button('删除', { type: ButtonType.Normal, stateEffect: true })
          .width(80)
          .height(40)
          .margin({ right: 20 })
          .borderRadius(8)
          .backgroundColor('#f75653')
          .onClick(() => {
            console.log('删除这一步骤')
          })
      }
      .width('100%')
      .margin({ bottom: 10 })
      TextArea({ placeholder: '具体操作' }).fontSize(16)
    }
    .margin({ top: 10, bottom: 10 })
    .alignItems(HorizontalAlign.Start)
  }
}

@Entry
@Component
struct Main {
  build() {
    Column() {
      PageTitle()
      DishOutline()
      List() {
        ListItem() {
          CookingStep()
        }
        ListItem() {
          CookingStep()
        }
        ListItem() {
          CookingStep()
        }
      }
      .width(100)
      .height(100)
      Button('下一步', { type: ButtonType.Normal, stateEffect: true })
        .width(80)
        .height(40)
        .margin({ left: 30, top: 10 })
        .borderRadius(8)
        .backgroundColor('#8ce559')
        .onClick(() => {
          console.log('新增一个步骤文本输入')
        })
    }
    .alignItems(HorizontalAlign.Start)
  }
}