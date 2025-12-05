
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
struct NavigationApi1 {
  @Provide('pageInfos') pageInfos: NavPathStack = new NavPathStack()
  finishTransitionTest:string = '111'
  customNavContentTransitionTest: string = '333'
  @State x: number = 0
  scaleVal: number = 1
  @State indexes:Array<number> = new Array<number>()
  @State result:string = ''
  index:string = ''
  @State text2:string = 'none'
  @State result1: string = '';
  popResult2: string = '';
  message:string = ''



  @Builder pageOneTmp() {
    NavDestination() {
      Column() {
        Button('pushPathByName', { stateEffect: true, type: ButtonType.Capsule })
          .width('80%')
          .height(40)
          .margin(20)
          .id('pushPathByName_test')
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
        //pageOne
        this.text2 = JSON.parse(JSON.stringify(ctx.pathInfo)).name
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
      Text(this.text2).fontColor(Color.Red).fontSize(14).id('NavigationText')
      Text(this.result).fontColor(Color.Yellow).fontSize(14).id('removeText')
      Text(this.result1).fontColor(Color.Green).fontSize(14).id('deleteText')


      Button('点击删除索引在Indexes的页面').onClick(() => {
        let ret = this.pageInfos.removeByIndexes(this.indexes)
        this.result = '' + ret
      }).id('removeIndexes').margin({bottom: 10})

      Button('点击删除Name的页面').onClick(() => {
        let ret1 = this.pageInfos.removeByName('pageTwo')
        this.result1 = '' + ret1
      }).margin({bottom: 10}).id('deleteName')

      Navigation(this.pageInfos) {
      }.title('NavIndex').navDestination(this.PageMap)
      .hideNavBar(true)

    }
  }
}