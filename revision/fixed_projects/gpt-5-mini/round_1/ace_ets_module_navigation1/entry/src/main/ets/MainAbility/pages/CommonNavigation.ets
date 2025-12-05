import events_emitter from '@ohos.events.emitter'

@Component
struct PageOne {
  @State tag:string = ""
  @State param:string = ""

  build() {
    NavDestination() {
      Text(this.tag)
    }.title('pageOne Standard')
    .onActive((reason: NavDestinationActiveReason) => {
      this.tag = reason + '';
      console.log(`${this.tag} pageOne is onActive: ${reason}`);
      console.log(`${this.tag} pageOne param is : ${this.param}`);
      AppStorage.setOrCreate('onActive', reason);
    }).onInactive((reason: NavDestinationActiveReason) => {
      this.tag = reason + ''
      console.log(`${this.tag} pageOne is onInActive: ${reason}`);
      AppStorage.setOrCreate('onInactive', reason);
    })
    .onReady((context) => {
      this.param = context?.pathInfo?.param as string;
      console.log(`onReady pageOne is param: ${this.param}`)
    })
  }
}

@Component
export struct CommonNavigation {
  pathStack?: NavPathStack
  title:string = ''

  @Builder
  PageBuilder(name: string, param: string) {
    if (name == 'pageOne') {
      PageOne({tag: param})
    }
  }

  build() {
    Navigation(this.pathStack) {
    }.title(this.title)
    .navDestination(this.PageBuilder)
  }
}