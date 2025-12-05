@Entry
@Component
struct PathStackPage {
  @Provide('mainPathStack') stack: NavPathStack = new NavPathStack();

  @Builder
  MyPageMap(name: string) {
    if (name === 'navPathPageOne') {
      NavPathPageOne()
    } else if (name === 'navPathPageTwo') {
      NavPathPageTwo()
    }
  }


  build() {
    Navigation(this.stack) {
      Button('pushStack').onClick(() => {
        let curPathStack: Array<NavPathInfo> = new Array();

        let popCallbackOne: Callback<PopInfo> = (popupInfo: PopInfo) => {
          console.log('pop setPathStack pageOne' + JSON.stringify(popupInfo.info))
        }
        let popCallbackTwo: Callback<PopInfo> = (popupInfo: PopInfo) => {
          console.log('pop setPathStack pageTwo' + JSON.stringify(popupInfo.info))
        }
        curPathStack.push(new NavPathInfo('navPathPageOne', 'setPathStack pageOne', popCallbackOne, false), new NavPathInfo('navPathPageTwo', 'setPathStack pageTwo', popCallbackTwo, true));
        this.stack.setPathStack(curPathStack, true);
      })
        .id('pushId')

    }
    .width('100%')
    .height('100%')
    .title('MainPage')
    .navDestination(this.MyPageMap)
  }
}

@Component
struct NavPathPageOne {
  private stack: NavPathStack | undefined = undefined;
  build() {
    NavDestination() {
    }
    .width('100%').height('100%')
    .title('navPathPageOne')
  }
}

@Component
struct NavPathPageTwo {
  @Consume('mainPathStack') pathStack: NavPathStack
  build() {
    NavDestination() {
      Button('getStack').onClick(() => {
        let allStacks =  this.pathStack.getPathStack();
        console.log('all stack is: ', allStacks?.length)
        AppStorage.setOrCreate('stackLen', allStacks?.length)
        console.log('all stack[0] navDestinationId is', allStacks[1]?.navDestinationId)
        AppStorage.setOrCreate('navDestinationId1', Number(allStacks[1]?.navDestinationId))
      })
        .id('getId')
    }
    .width('100%').height('100%')
    .title('navPathPageTwo')
  }
}