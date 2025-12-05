
/*
 * Copyright (c) 2025 Huawei Device Co., Ltd.
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
// Index.ets


@Entry
@Component
struct NavigationExample {
  @Provide('pageInfos') pageInfos: NavPathStack = new NavPathStack();
  @Builder
  pageMap(name:string){
    if(name == 'pageOne'){
      PageOneInterception()
    }else{
      PageTwo199()
    }
  }
  registerInterception() {
    this.pageInfos.setInterception({
      interception: (from: NavPathInfo | 'navBar', to: NavPathInfo | 'navBar', navStack: NavPathStack,
        operation: NavigationOperation, animated: boolean) => {
        console.log('aaa popToIndex NavStack:' + navStack.getAllPathName())
        if (typeof to === 'string') {
          return;
        }

        let target: NavPathInfo = to as NavPathInfo;
        if (target.name === 'pageTwo') {
          navStack.pop();
          navStack.pushPathByName('pageOne', null)
        }
        console.log('aaa popToIndex NavStack:' + navStack.getAllPathName())
      },
      willShow: (from: NavDestinationContext | 'navBar', to: NavDestinationContext | 'navBar',
        operation1: NavigationOperation, animated: boolean) => {
        let target: NavDestinationContext = to as NavDestinationContext;
        target.pathStack.pushPathByName('pageTwo',null)
      },
      didShow: (from: NavDestinationContext | 'navBar', to: NavDestinationContext | 'navBar',
        operation: NavigationOperation, isAnimated: boolean) => {
      },
      modeChange: (mode: NavigationMode) => {
        console.info(`current navigation mode is ${mode}`);
      }
    })
  }

  build() {
    Navigation(this.pageInfos) {
      Column() {
        Button('push pageTwo', { stateEffect: true, type: ButtonType.Capsule })
          .id('PageId')
          .width('80%')
          .height(40)
          .margin(20)
          .onClick(() => {
            this.registerInterception();
            this.pageInfos.pushPath({ name: 'pageTwo' });
          })
      }
    }.title('NavIndex').navDestination(this.pageMap)
  }
}

@Component
struct PageOneInterception {
  @Consume('pageInfos') pageInfos: NavPathStack;

  build() {
    NavDestination() {
      Text('This is page one')
      Button('button').id('PageOne')
    }
    .title('pageOne')
    .onReady(() => {
      console.log('aaa pageOne this.pageInfos ', this.pageInfos.getAllPathName())
    })
  }
}

@Component
struct PageTwo199 {
  @Consume('pageInfos') pageInfos: NavPathStack;
  @State text: string = 'text';
  build() {
    NavDestination() {
      Text('This is page two')
      Text(this.text).id('PageTwoId')
    }
    .title('pageTwo')
    .onReady(() => {
      this.text = this.pageInfos.getAllPathName().toString()
    })
  }
}