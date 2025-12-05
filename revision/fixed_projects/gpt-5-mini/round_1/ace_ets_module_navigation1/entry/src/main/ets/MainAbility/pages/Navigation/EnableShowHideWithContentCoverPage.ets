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
struct EnableShowHideWithContentCoverPage {
  @State enableValue: boolean = true;

  private stack: NavPathStack = new NavPathStack();

  @Builder
  pageMap(name: string) {
    if (name == 'pageOne') {
      PageOne22()
    }
  }

  build() {
    Navigation(this.stack){
      Button('changeEnableValue').onClick(() => {
        this.enableValue = !this.enableValue;
      })
      Button('pushOne').onClick(() => {
        this.stack.pushPath({name: 'pageOne'})
      })
    }
    .titleMode(NavigationTitleMode.Mini)
    .title('Navigation')
    .enableVisibilityLifecycleWithContentCover(this.enableValue)
    .navDestination(this.pageMap)
    .mode(NavigationMode.Stack)

  }
}

@Component
struct PageOne22 {
  private stack: NavPathStack | undefined = undefined;
  @State isShow: boolean = false;
  @State testModal: boolean = false;

  @Builder
  myBuilder() {
    Flex({alignItems: ItemAlign.Center, justifyContent: FlexAlign.Center}){
      Button('closeModal')
        .onClick(() => {
          this.isShow = false;
        })
    }
    .backgroundColor('#ccc')
    .width('100%')
    .height('100%')
  }

  build() {
    NavDestination() {
      Button('back').onClick(() => {
        this.stack?.pop()
        this.testModal = false;
      })
      Button('openModal')
        .onClick(() => {
          AppStorage.delete('testValue')
          this.testModal = true;
          this.isShow = true;
        })
        .bindContentCover(
          this.isShow,
          this.myBuilder(),
          {
            modalTransition: ModalTransition.DEFAULT,
            backgroundColor: Color.Pink,
            transition: TransitionEffect.asymmetric(
              TransitionEffect.OPACITY.animation({ duration: 500 }).combine(
                TransitionEffect.rotate({ z: 1, angle: 180 }).animation({ delay: 50, duration: 500 }))
              ,
              TransitionEffect.OPACITY.animation({ duration: 500 }).combine(
                TransitionEffect.rotate({ z: 1, angle: 180 }).animation({ duration: 500 }))
            ),
            onWillDismiss: ((dismissContentCoverAction: DismissContentCoverAction) => {
              if (dismissContentCoverAction.reason == DismissReason.PRESS_BACK) {
                console.log('back pressed');
              }
              dismissContentCoverAction.dismiss();
            }),
            onAppear: () => {
              console.log('BindContentCover onAppear.');
            },
            onDisappear: () => {
              this.isShow = false;
              console.log('BindContentCover onDisappear.');
            }
          })
    }.title('PageOne')
    .onReady((ctx: NavDestinationContext) => {
      this.stack = ctx?.pathStack;
    })
    .onShown((callback) => {
      if (this.testModal) {
        console.log('EnableShowHideWithContentCoverPage show name: ', callback as VisibilityChangeReason == VisibilityChangeReason.CONTENT_COVER);
      } else {
        console.log('EnableShowHideWithContentCoverPage show name: ', callback as VisibilityChangeReason == VisibilityChangeReason.TRANSITION);
      }
      console.log('EnableShowHideWithContentCoverPage show name: ', JSON.stringify(callback as VisibilityChangeReason))
      AppStorage.setOrCreate('testValue', JSON.stringify(callback as VisibilityChangeReason));
    })
    .onHidden((callback) => {
      if (this.testModal) {
        console.log('EnableShowHideWithContentCoverPage hide name: ', callback as VisibilityChangeReason == VisibilityChangeReason.CONTENT_COVER);
      } else {
        console.log('EnableShowHideWithContentCoverPage hide name: ', callback as VisibilityChangeReason == VisibilityChangeReason.TRANSITION);
      }
      console.log('EnableShowHideWithContentCoverPage hide name: ', JSON.stringify(callback as VisibilityChangeReason))
      AppStorage.setOrCreate('testValue', JSON.stringify(callback as VisibilityChangeReason));
    })
  }
}