/**
 * Copyright (c) 2024 Huawei Device Co., Ltd.
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
import {CustomTransition} from './CustomNavigationUtils'

@Component
export struct PageTwoTemp {
  @Consume('pageInfos') pageInfos: NavPathStack
  @State x: number = 300

  aboutToAppear() {

    CustomTransition.getInstance().registerNavParam('pageTwo', (isPush: boolean, isExit: boolean)=>{
      console.log("current page is pageOne")
      this.x = isExit ? 0 : 300;
    }, (isPush: boolean, isExit: boolean)=>{
      this.x = isExit ? -300 : 0;
    }, 200)
  }

  aboutToDisappear() {
    CustomTransition.getInstance().unRegisterNavParam('pageOne')
  }

  build() {
    NavDestination() {
      Column() {
        Button('pushPathByName', { stateEffect: true, type: ButtonType.Capsule })
          .width('80%')
          .height(40)
          .margin(20)
          .onClick(() => {
            this.pageInfos.pushPathByName('pageOne', null) //将name指定的NavDestination页面信息入栈，传递的数据为param
          })
      }.width('100%').height('100%')
    }.title('pageTwo')
    .backgroundColor(Color.White)
    .onBackPressed(() => {
      const popDestinationInfo = this.pageInfos.pop() // 弹出路由栈栈顶元素
      console.log('pop' + '返回值' + JSON.stringify(popDestinationInfo))
      return true
    })
    .translate({x: this.x})
    .backgroundColor(Color.White)
  }
}