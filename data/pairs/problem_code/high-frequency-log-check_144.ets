import hilog from '@ohos.hilog';
@Entry
@Component
struct App {
  build() {
    Toggle()
      .onActionUpdate(() => {
        hilog.debug(3001, 'App', 'Toggle Action Update')
    })
  }
}