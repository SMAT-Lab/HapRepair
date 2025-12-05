
import UIContext from '@ohos.arkui.UIContext';
import observer from '@ohos.arkui.observer';

@Component
struct PageOne {
  @Consume('stackA') stackA:NavPathStack;
  build() {
    NavDestination() {
      Text('pageOne')
      Button('back').onClick(() => {
        this.stackA.pop();
      })
      .height('5%')
      .id('backId')
    }.title('pageOne')
  }
}

@Entry
@Component
struct NavDestinationUIObserver {
  @Provide stackA: NavPathStack = new NavPathStack();
  stackB: NavPathStack = new NavPathStack();
  private obs: UIContext.UIObserver | null = null;
  showText: string = '';

  @Builder
  PageMapA(name: string) {
    if (name === 'pageOne') {
      PageOne()
    } else if (name === 'pageTwo'){
      PageOne()
    } else if (name === 'pageThree'){
      PageOne()
    }
  }

  @Builder
  PageMapB(name: string) {
    if (name === 'pageOne') {
      PageOne()
    }
  }

  navDestinationSwitchCallbackWithUIContext1(info: observer.NavDestinationSwitchInfo) {
    if (info) {
      if (info.context) {
        AppStorage.setOrCreate('showText', 'on');
        console.log(`testTag navDestinationSwitchCallbackWithUIContext1, info: \"from\":${typeof info.from === 'string' ? ('"' + info.from + '"') : JSON.stringify(info.from)},\"to\":${typeof info.to === 'string' ? ('"' + info.to + '"') : JSON.stringify(info.to)}`);
      }
    }
  }

  navDesUpdateCallback1(info: NavDestinationInfo) {
    AppStorage.setOrCreate('showUpdateText', 'on');
    console.log(`testTag navDesUpdateCallback1, info: ${AppStorage.get('showUpdateText') as string}`);
  }

  aboutToAppear(): void {
    this.obs = this.getUIContext().getUIObserver();
  }

  contextObserverOn(){
    this.obs?.on('navDestinationSwitch', this.navDestinationSwitchCallbackWithUIContext1);
    this.obs?.on('navDestinationUpdate', this.navDesUpdateCallback1);
  }

  contextObserverOff(){
    AppStorage.setOrCreate('showText', 'off');
    AppStorage.setOrCreate('showUpdateText', 'off');
    console.log(`testTag navDesUpdateCallback1, info: ${AppStorage.get('showUpdateText') as string}`);
    this.obs?.off('navDestinationSwitch', this.navDestinationSwitchCallbackWithUIContext1);
    this.obs?.off('navDestinationUpdate', this.navDesUpdateCallback1);
  }

  build() {
    Column({space:5}) {
      Button('contextObserver.on')
        .onClick(() => {
          this.contextObserverOn()
        })
        .height('5%')
        .id('onBtn')
        .margin({top: '2%'})

      Button('contextObserver.off')
        .onClick(() => {
          this.contextObserverOff()
        })
        .height('5%')
        .id('offBtn')

      Text(this.showText).key('showTextKey').height('5%')
      Navigation(this.stackA) {
        Column({space:10}){
          Button('MainA push pageOne')
            .onClick(() => {
              this.stackA.pushPath({name: 'pageOne', param: 1})
            })
            .id('pushId')
        }
      }
      .navDestination(this.PageMapA)
      .title('MainA')
      .titleMode(NavigationTitleMode.Mini)
      .backgroundColor(Color.Pink)
      .width('100%')
    }
    .width('100%')
    .height('100%')
  }
}