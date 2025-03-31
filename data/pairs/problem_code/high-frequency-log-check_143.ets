import hilog from '@ohos.hilog';
@Entry
@Component
struct App {
  build() {
    ListItem()
      .onMouse(() => {
        hilog.info(2001, 'App', 'onMouse')
    })
  }
}