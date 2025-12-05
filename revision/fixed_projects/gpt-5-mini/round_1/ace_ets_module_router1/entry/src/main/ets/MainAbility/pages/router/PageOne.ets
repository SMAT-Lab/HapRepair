import { router } from '@kit.ArkUI'
import events_emitter from '@ohos.events.emitter';

class RouteParam {
  constructor(msg: string) {
    this.msg = msg
  }
  private msg: string = '';
}

@Entry({routeName: 'nickNameOne'})
@Component
struct PageOne {

  aboutToAppear(): void {
    console.log('PageOne aboutToAppear');
  }

  onPageShow(): void {
    console.log('PageOne onPageShow');
    this.queryRouterPageInfo()
  }

  onNewParam(param: ESObject) {
    console.log('PageOne onNewParam, param: ' + JSON.stringify(param));
    let backData: events_emitter.EventData = {
      data: {
        'paramString': JSON.stringify(param)
      }
    }

    let eventIdNum = 51;
    if (JSON.stringify(param) === '"test string"') {
      eventIdNum = 52;
    } else if (JSON.stringify(param) === '20') {
      eventIdNum = 53;
    }
    let backEvent: events_emitter.InnerEvent = {
      eventId: eventIdNum,
      priority: events_emitter.EventPriority.LOW
    }
    console.info('onNewParam start to emit action state')
    events_emitter.emit(backEvent, backData)
  }

  build() {
    Navigation() {
      Column() {
        Button('pushUrl PageOne Single')
          .onClick(() => {
            router.pushUrl({ url: 'MainAbility/pages/router/PageOne', params: new RouteParam('pushUrl One Single') }, router.RouterMode.Single)
          })
          .id('pushUrlId')
          .height('10%')
        Button('replaceUrl PageOne Single')
          .onClick(() => {
            router.replaceUrl({ url: 'MainAbility/pages/router/PageOne', params: 'test string'}, router.RouterMode.Single)
          })
          .id('replaceUrlId')
          .height('10%')
        Button('pushNamed PageOne Single')
          .onClick(async () => {
            await import ('./PageOne')
              .then(() => {
                router.pushNamedRoute({ name: 'nickNameOne', params: 20 }, router.RouterMode.Single)
              })
          })
          .id('pushNamedId')
          .height('10%')
      }
      .width('100%')
      .height('100%')
    }
    .title('PageOne')
    .titleMode(NavigationTitleMode.Mini)
  }
}