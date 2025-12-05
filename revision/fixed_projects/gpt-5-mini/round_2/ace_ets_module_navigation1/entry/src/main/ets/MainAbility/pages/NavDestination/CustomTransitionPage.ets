/*
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

declare type voidFunc = () => void;

@Entry
@Component
struct CustomTransitionPage {
  @Provide('pageInfos') pageInfos: NavPathStack = new NavPathStack()

  @Provide('pushTrue') pushTrue: number = 0;
  @Provide('pushFalse') pushFalse: string = '';

  @Provide('popTrue') popTrue: string = '';
  @Provide('popFalse') popFalse: string = '';

  @Provide('replaceTrue') replaceTrue: string = '';
  @Provide('replaceFalse') replaceFalse: string = '';

  @Builder
  pageMap(name: string) {
    if (name == 'pageOne') {
      PageOne77()
    } else if (name == 'pageTwo') {
      PageTwo77()
    }
  }

  build() {
    Column() {
      Column({space: '5vp'}) {
        Text(this.pushTrue + '').id('pushTrueId').height('5%')
        Text(this.pushFalse).id('pushFalseId').height('5%')
        Text(this.popTrue).id('popTrueId').height('5%')
        Text(this.popFalse).id('popFalseId').height('5%')
        Text(this.replaceTrue).id('replaceTrueId').height('5%')
        Text(this.replaceFalse).id('replaceFalseId').height('5%')
      }
      Navigation(this.pageInfos) {
        Column({space: '5vp'}) {
          Button('push pageOne').id('pushOneId').onClick(() => {
            this.pageInfos.pushPath({name: 'pageOne'} )
          }).height('10%')
        }
        .width('100%')
        .height('100%')
      }
      .title('Title')
      .titleMode(NavigationTitleMode.Mini)
      .navDestination(this.pageMap)
      .mode(NavigationMode.Stack)
    }
  }
}

@Component
struct PageOne77 {
  @Consume('pageInfos') pageInfos: NavPathStack;

  @Consume('pushTrue') pushTrue: number;
  @Consume('pushFalse') pushFalse: string;

  @Consume('popTrue') popTrue: string;
  @Consume('popFalse') popFalse: string;

  @Consume('replaceTrue') replaceTrue: string;
  @Consume('replaceFalse') replaceFalse: string;

  build() {
    NavDestination() {
      Button('push pageTwo').id('pushId').onClick(() => {
        this.pageInfos.pushPath({name: 'pageTwo'} )
      }).height('10%')
    }
    .title('pageOne')
    .customTransition((op: NavigationOperation, isEnter: boolean): Array<NavDestinationTransition> | undefined => {
      if (op === NavigationOperation.PUSH) {
        if (isEnter) {
          this.pushTrue = 50
        } else {
          this.pushFalse = 'push_0';
        }
      } else if (op === NavigationOperation.POP) {
        if (isEnter) {
          this.popTrue = 'pop_1'
        } else {
          this.popFalse = 'pop_0';
        }
      } else if (op === NavigationOperation.REPLACE) {
        if (isEnter) {
          this.replaceTrue = 'replace_1';
        } else {
          this.replaceFalse = 'replace_0'
        }
      }
	  return undefined;
	})
  }
}

@Component
struct PageTwo77 {
  @Consume('pageInfos') pageInfos: NavPathStack;

  @Consume('pushTrue') pushTrue: number;
  @Consume('pushFalse') pushFalse: string;

  @Consume('popTrue') popTrue: string;
  @Consume('popFalse') popFalse: string;

  @Consume('replaceTrue') replaceTrue: string;
  @Consume('replaceFalse') replaceFalse: string;
  build() {
    NavDestination() {
      Button('pop').id('popId').onClick(() => {
        this.pageInfos.pop()
      }).height('10%')
      Button('replace pageOne').id('replaceId').onClick(() => {
        this.pageInfos.replacePath({name: 'pageOne'} )
      }).height('10%')
    }
    .title('pageTwo')
    .customTransition((op: NavigationOperation, isEnter: boolean): Array<NavDestinationTransition> | undefined => {
      let transitionOneDuration: number = 500;
      let testTag: string = '[NavDestCustomTransition]';
      let transitionOneEvent: voidFunc = () => { console.log(testTag, 'reached transitionOne, empty now!'); }
      let transitionOneFinishEvent: voidFunc = () => {
      }
      if (op === NavigationOperation.PUSH) {
        if (isEnter) {
          transitionOneEvent = () => {
            console.log(testTag, 'transitionOne, push & isEnter');
            this.pushTrue = 100;
          }
        } else {
          transitionOneEvent = () => {
            console.log(testTag, 'transitionOne, push & !isEnter');
            this.pushFalse = 'push_0';
          }
          transitionOneDuration = 450
        }
      } else if (op === NavigationOperation.POP) {
        console.log('pop op is: ', op)
        console.log('pop isEnter is: ', isEnter)
        if (isEnter) {
          transitionOneEvent = () => {
            console.log(testTag, 'transitionOne, pop & isEnter');
            this.popTrue = 'pop_1'
          }
        } else {
          transitionOneEvent = () => {
            console.log(testTag, 'transitionOne, pop & !isEnter');
            this.popFalse = 'pop_0';
          }
        }
      } else if (op === NavigationOperation.REPLACE) {
        console.log('replace op is: ', op)
        console.log('replace isEnter is: ', isEnter)
        if (isEnter) {
          transitionOneEvent = () => {
            console.log(testTag, 'transitionOne, replace & isEnter');
            this.replaceTrue = 'replace_1';
          }
        } else {
          transitionOneEvent = () => {
            console.log(testTag, 'transitionOne, replace & !isEnter');
            this.replaceFalse = 'replace_0'
          }
        }
      } else {
        console.log(testTag, '----- NOT-IMPL BRANCH of NAV-DESTINATION CUSTOM TRANSITION -----');
      }
      let transitionOne: NavDestinationTransition = {
        duration: transitionOneDuration,
        delay: 0,
        curve: Curve.Friction,
        event: transitionOneEvent,
        onTransitionEnd: transitionOneFinishEvent
      };
      return [
        transitionOne
      ];
    })
  }
}
