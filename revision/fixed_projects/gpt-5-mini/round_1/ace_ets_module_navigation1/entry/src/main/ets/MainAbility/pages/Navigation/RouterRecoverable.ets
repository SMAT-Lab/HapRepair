import { router } from '@kit.ArkUI';

class InnerParams {
  private data3: number[]

  constructor(tuple:number[]) {
    this.data3 = tuple
  }
}
class RouterParams {
  private data1: string
  private data2: InnerParams

  constructor(str:string, tuple:number[]) {
    this.data1 = str
    this.data2 = new InnerParams(tuple)
  }
}
@Entry
@Component
struct RouterRecoverable {
 
 
 
  controller: TextInputController = new TextInputController()
  routerPageUpdateCallback(info: RouterPageInfo) {
    if(info){
       console.log(`testTag routerPageUpdateCallback, index: ${info.index}, name: ${info.name}, path: ${info.path}, state: ${info.state}, pageId: ${info.pageId}`);
    }
  }

  build() {
    Column({ space: '20vp' }) {
      Button('routerPage2_recoverable=true', { stateEffect: true, type: ButtonType.Capsule })
        .id('routerBtnId1')
        .width('80%')
        .height(40)
        .margin({ bottom: 10 })
        .onClick(()=>{
          router.pushUrl({
            url: 'MainAbility/pages/Navigation/RouterPage2',
            params: new RouterParams('message' ,[123,456,789]),
            recoverable:true
          })
        })
      Button('routerPage2_recoverable=false', { stateEffect: true, type: ButtonType.Capsule })
        .id('routerBtnId2')
        .width('80%')
        .height(40)
        .margin({ bottom: 10 })
        .onClick(()=>{
          router.pushUrl({
            url: 'MainAbility/pages/Navigation/RouterPage2',
            params: new RouterParams('message' ,[123,456,789]),
            recoverable:false
          })
        })
    }
    .height('100%')
    .width('100%')
    .margin({top: '20vp'})
  }
}