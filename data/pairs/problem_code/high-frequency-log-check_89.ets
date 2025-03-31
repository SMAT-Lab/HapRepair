import hilog from '@ohos.hilog';
@Entry
@Component
struct MainScreen {
  build() {
    Swipe()
      .onChange(() => {
        hilog.debug(9005, 'MainScreen', 'onChange')
      })
  }
}