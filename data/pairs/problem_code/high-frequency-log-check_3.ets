import hilog from '@ohos.hilog';
@Entry
@Component
struct Index {
  build() {
    Button("")
      .width(200)
      .height(100)
      .onMouse(() => {
        hilog.info(1001, 'Index', 'onMouse')
      })
  }
}