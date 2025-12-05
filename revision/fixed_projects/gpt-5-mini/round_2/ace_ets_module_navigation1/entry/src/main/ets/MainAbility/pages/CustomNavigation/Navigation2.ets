
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
import { CustomTransition, AnimateCallback } from './common/CustomNavigationUtils'
import {PageTwoTemp} from './common/PageTwo'
@Entry
@Component
struct NavigationApi3 {
  @Provide('pageInfos') pageInfos: NavPathStack = new NavPathStack()
  finishTransitionTest:string = '111'
  customNavContentTransitionTest: string = '333'
  @State x: number = 0
  scaleVal: number = 1
  indexes:Array<number> = new Array<number>()
  index:string = ''
  @State text2:string = 'none'


  @Builder pageOneTmp() {
    NavDestination() {
      Column() {
        Text(this.text2).fontColor(Color.Red).fontSize(14).id('NavigtionText2')
      }.width('100%').height('100%')
    }.title('pageOne')
    .mode(NavDestinationMode.STANDARD)
    .onBackPressed(() => {
      const popDestinationInfo = this.pageInfos.pop() // 弹出路由栈栈顶元素
      console.log('pop' + '返回值' + JSON.stringify(popDestinationInfo))
      return true
    })
    .onReady((ctx: NavDestinationContext) => {
      try {
        //NavPathStack
        this.text2 = JSON.parse(JSON.stringify(ctx.pathStack)).type
      } catch(e) {
      }
    })
    .translate({x: this.x, y: 0, z: 0})
    .backgroundColor(Color.White)
  }

  @Builder
  PageMap(name: string) {
    if (name === 'pageOne') {
      this.pageOneTmp()
    } else if (name === 'pageTwo') {
      PageTwoTemp()
    }
  }

  aboutToAppear() {
    CustomTransition.getInstance().registerNavParam('pageOne', (isPush: boolean, isExit: boolean) => {
      this.x = isExit ? 0 : 300;
    }, (isPush: boolean, isExit: boolean)=> {
      this.x = isExit ? -300 : 0;
    }, 200);
    if (this.pageInfos === undefined) {
      this.pageInfos = new NavPathStack();
    }
    this.pageInfos.pushPath({name: 'pageOne'}, false)

  }

  aboutToDisappear() {
    CustomTransition.getInstance().unRegisterNavParam('pageOne')
  }
  build() {
    Column() {
      Navigation(this.pageInfos) {
      }.title('NavIndex').navDestination(this.PageMap)
      .hideNavBar(true)

    }
  }
}