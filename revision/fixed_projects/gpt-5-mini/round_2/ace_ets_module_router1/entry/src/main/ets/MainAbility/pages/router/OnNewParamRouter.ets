import { router } from '@kit.ArkUI'

@Entry
@Component
struct OnNewParamRouter {
  build() {
    Navigation() {
      Column() {
        Button('push pageOne')
          .onClick(() => {
            router.pushUrl({ url: 'MainAbility/pages/router/PageOne'})
          })
          .id('pushId')
          .height('10%')
      }
    }
    .title('MainPage')
    .hideBackButton(true)
    .titleMode(NavigationTitleMode.Mini)
    .mode(NavigationMode.Stack)
  }
}

