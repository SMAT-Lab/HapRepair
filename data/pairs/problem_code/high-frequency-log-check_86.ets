import hilog from '@ohos.hilog';
@Entry
@Component
struct Dashboard {
  build() {
    Scroll()
      .onScrollStart(() => {
        hilog.info(6002, 'Dashboard', 'onScroll')
      })
  }
}