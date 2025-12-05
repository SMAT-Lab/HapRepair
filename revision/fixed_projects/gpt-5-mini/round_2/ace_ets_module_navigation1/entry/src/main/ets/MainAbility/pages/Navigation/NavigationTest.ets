
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

@Entry
@Component
struct NavigationExample2 {
  @Provide('pageInfos') pageInfos: NavPathStack = new NavPathStack();
  private isUseInterception: boolean = true;
  // interceptionText 未发生变化，改为普通变量
  interceptionText: string = '路由拦截：启用';
  @State logText: string = 'Navigation 日志: \n';
  private updateLog(content: string) {
    this.logText = `${content}`;
    console.info(`PageOne001 Log: ${content}`);
  }

  @Builder
  pageMap(name:string){
    if(name == 'pageOne001'){
      PageOne001()
    }else{
      PageTwo002()
    }
  }

  private registerInterception() {
    this.pageInfos.setInterception({
      willShow: (
        from: NavDestinationContext | NavBar,
        to: NavDestinationContext | NavBar
      ) => {
        if (!this.isUseInterception) {
          return;
        }
        
        if (typeof from === 'string' || typeof to === 'string') {
          return;
        }

        let target: NavDestinationContext = to as NavDestinationContext;
        const logContent = `willShow:from.mode ${from.mode}to.mode ${to.mode}`;
        this.updateLog(logContent);

        if (target.pathInfo.name === 'pageTwo002') {
          target.pathStack.pop();
          target.pathStack.pushPathByName('pageOne001', null);
        }
      },
      didShow: (
        from: NavDestinationContext | NavBar,
        to: NavDestinationContext | NavBar
      ) => {
        if (!this.isUseInterception) {
          return;
        }
        if (typeof from === 'string' || typeof to === 'string') {
          return;
        }

        let target: NavDestinationContext = to as NavDestinationContext;
        const logContent = `didShow:from.mode ${from.mode}to.mode ${to.mode}`;
        this.updateLog(logContent);
        if (target.pathInfo.name === 'pageTwo002') {
          target.pathStack.pop();
          target.pathStack.pushPathByName('pageOne001', null);
        }
      },
    })
  }

  build() {
    Navigation(this.pageInfos) {
       Column() {
          Button('pushPath pageTwo002', { stateEffect: true, type: ButtonType.Capsule })
            .id('pageTwo002')
            .width('80%')
            .height(40)
            .margin(10)
            .onClick(() => {
              this.pageInfos.pushPath({ name: 'pageTwo002' });
            })
          Button('pushPath pageOne001', { stateEffect: true, type: ButtonType.Capsule })
            .id('pageOne001')
            .width('80%')
            .height(40)
            .margin(10)
            .onClick(() => {
              this.pageInfos.pushPath({ name: 'pageOne001' });
            })
          Text(this.interceptionText)
            .fontSize(10)
            .fontColor(Color.Green)
        }
        .width('100%')
        .height('100%')
      Column({ space: 15 }) {
        Text(this.logText)
          .width('100%')
          .backgroundColor('#fffcf0f0')
          .fontSize(10)
          .id('indexLogId')
      }
      .width('100%')
      .height('100%')
    }
    .navDestination(this.pageMap)
    .onAppear(() => {
      this.registerInterception();
      this.updateLog('路由拦截：开启');
    })
  }
}
class TmpClass {
  private count: number = 10;
}
@Component
struct PageOne001 {
  @Consume('pageInfos') pageInfos: NavPathStack;
  @State logText: string = '日志: \n';
  private updateLog(content: string) {
    this.logText = `日志: \n${content}`;
    console.info(`PageOne001 Log: ${content}`);
  }
  build() {
    NavDestination() {
      Column({ space: 15 }) {
        Column() {
          Button('push pageTwo002', { stateEffect: true, type: ButtonType.Capsule })
            .id('pageTwo02')
            .width('80%')
            .height(40)
            .margin(20)
            .onClick(() => {
              let tmp = new TmpClass();
              this.pageInfos.pushPathByName('pageTwo002', tmp);
              this.updateLog('点击pushPathByName：跳转至pageTwo002');
            })
        }
        .width('100%')
        .height('100%')

        Text(this.logText)
          .width('100%')
          .backgroundColor('#ffffe6e6')
          .fontSize(10)
          .id('pageOne001LogId')

      }
      .width('100%')
      .height('100%')
    }
    .title('pageOne001')
    .mode(NavDestinationMode.STANDARD)
    .onBackPressed(() => {
      const popDestinationInfo = this.pageInfos.pop();
      const logContent = `点击返回键：回退页面：${JSON.stringify(popDestinationInfo)}）`;
      this.updateLog(logContent);
      console.info(JSON.stringify(popDestinationInfo));
      return true;
    })
    .onReady((context: NavDestinationContext) => {
      this.pageInfos = context.pathStack;
      this.updateLog('pageOne001 mode：STANDARD');
    })
  }
}

@Component
struct PageTwo002 {
  @Consume('pageInfos') pageInfos: NavPathStack;
  @State logText: string = '日志: \n';
  private updateLog(content: string) {
    this.logText = `日志: \n${content}`;
    console.info(`PageTwo002 Log: ${content}`);
  }

  build() {
    NavDestination() {
      Column({ space: 15 }) {

        Column() {
          Button('push pageOne001', { stateEffect: true, type: ButtonType.Capsule })
            .id('pageOne01')
            .width('80%')
            .height(40)
            .margin(20)
            .onClick(() => {
              this.pageInfos.pushPathByName('pageOne001', null);
              this.updateLog('点击pushPathByName：跳转至pageOne');
            })
        }
        .width('100%')
        .height('100%')

        Text(this.logText)
          .width('100%')
          .backgroundColor('#ffffe6e6')
          .padding(12)
          .fontSize(14)
          .id('pageTwo002LogId')

      }
      .width('100%')
      .height('100%')
    }
    .title('pageTwo002')
    .mode(NavDestinationMode.DIALOG)
    .onBackPressed(() => {
      this.pageInfos.pop();
      this.updateLog('点击返回键：从pageTwo002页面回退');
      return true;
    })
    .onReady((context: NavDestinationContext) => {
      this.pageInfos = context.pathStack;
      this.updateLog('pageTwo002 mode：DIALOG');
    })
  }
}