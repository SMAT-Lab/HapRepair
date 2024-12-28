```arkts

/*
 * Copyright (C) 2023 Huawei Device Co., Ltd.
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

import promptAction from '@ohos.promptAction'
import router from '@ohos.router'
import socket from '@ohos.net.socket'
import Pop3Command, { Pop3LoginBean } from '@ohos/node-pop3'
import GlobalObj from '../GlobalObj'

@CustomDialog
struct CustomDialogDiy {
  @Link textValue: string
  @Link inputValue: string
  controller: CustomDialogController
  cancel: Function = () => {
  };
  confirm: Function = () => {
  };

  build() {
    Column() {
      Text('Please enter email type').fontSize(20).margin({ top: 10, bottom: 10 }).width('90%')
      TextInput({ placeholder: '', text: this.textValue }).height(60).width('90%')
        .onChange((value: string) => {
          this.textValue = value;
        })
      Text('Tips: Please enter the correct email type format, such as @qq.com or @163.com')
        .margin({ top: 10, bottom: 10 })
        .width('90%')
        .fontSize(8)
        .fontColor(Color.Red)
      Flex({ justifyContent: FlexAlign.SpaceAround }) {
        Button('Cancel')
          .onClick(() => {
            this.controller.close()
            this.cancel()
          })
        Button('Confirm')
          .onClick(() => {
            this.inputValue = this.textValue
            this.controller.close()
            this.confirm()
          })
      }
    }
  }
}

@Entry
@Component
struct Index {
  @State message: string = 'Hello World'
  @State account: string = 'xxx'
  @State password: string = 'xxx'
  @State mailType: string = '@qq.com'
  textValue: string = ''
  @State inputValue: string = 'click me'
  @State secure: boolean = false
  dialogController: CustomDialogController | null = new CustomDialogController({
    builder: CustomDialogDiy({
      cancel: () => {
        this.showToast(`The dialog box has been closed, and the option to enter a different type of email has been unselected`, 'mailType-cancel')
      },
      confirm: () => {
        this.mailType = this.inputValue
        this.showToast(`Enter other types of email addresses：${this.mailType}`, 'mailType-confirm')
      },
      textValue: $textValue,
      inputValue: $inputValue
    }),
    autoCancel: true,
    customStyle: false
  })

  aboutToDisappear() {
    this.dialogController = null;
    GlobalObj?.getInstance()?.getClient()?.QUIT();
  }

  showToast(text: string, name = 'test') {
    console.log(`zdy---${name}--->${text}`)
    promptAction.showToast({
      message: text,
      duration: 2000,
      bottom: 50
    })
  }

  @Builder
  MailMenu() {
    Menu() {
      MenuItem({ content: 'qq', labelInfo: 'qq' })
        .onChange((selected) => {
          if (selected) {
            this.mailType = '@qq.com'
          }
        })

      MenuItem({ content: '163', labelInfo: '163' })
        .onChange((selected) => {
          if (selected) {
            this.mailType = '@163.com'
          }
        })

      MenuItem({ content: '139', labelInfo: '139' })
        .onChange((selected) => {
          if (selected) {
            this.mailType = '@139.com'
          }
        })

      MenuItem({ content: 'sina', labelInfo: 'sina' })
        .onChange((selected) => {
          if (selected) {
            this.mailType = '@sina.com'
          }
        })
      MenuItem({ content: 'other', labelInfo: 'other' })
        .onChange((selected) => {
          if (selected) {
            if (this.dialogController) {
              this.dialogController.open()
            }
          }
        })
    }
  }

  build() {
    Row() {
      Column() {