
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

// PageOne.ets
import { BusinessError } from '@ohos.base';

class TmpClass{
  count:number = 10
}

class ParamWithOp {
  operation: number = 1
  count: number = 10
}

@Component
export struct PageOneTmp {
  @Consume('pageInfo') pageInfo: NavPathStack;
  @State message: string = 'Hello World'
  @State stack:NavPathStack = new NavPathStack()

  build() {
    NavDestination() {
      Column() {
        Text(this.message)
          .width('80%')
          .height(50)
          .margin(10)

        Button('pushPath', { stateEffect: true, type: ButtonType.Capsule })
          .width('80%')
          .height(40)
          .margin(10)
          .onClick(()=>{
            this.pageInfo.pushPath({name: 'pageTwo', param: new ParamWithOp(), onPop: (popInfo: PopInfo)=>{
              this.message = JSON.stringify(popInfo.result);
            }}); // 将name指定的NavDestination页面信息入栈，传递的数据为param，添加接收处理结果的onPop回调。
          }).id('pushPathButton')

        Button('get parent of current stack')
          .width('80%')
          .height(40)
          .margin(10)
          .onClick(()=>{
            let fatherStack:NavPathStack | null = this.stack.getParent()
            this.pageInfo.pushPath({name: 'pageTwo', param: new ParamWithOp(), onPop: (popInfo: PopInfo)=>{
              this.message = JSON.stringify(popInfo.result);
            }});
          }).id('getParent')

      }.width('100%').height('100%')
    }.title('pageOne')
    .onBackPressed(() => {
      this.pageInfo.pop({number: 1}) // 弹出路由栈栈顶元素。
      return true
    })
    .onReady((context: NavDestinationContext) => {
      this.stack = context.pathStack
      console.log('333' + this.stack)
    })
  }
}