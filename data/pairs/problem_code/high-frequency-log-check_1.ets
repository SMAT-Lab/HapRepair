import hilog from '@ohos.hilog';
@Entry
@Component
struct Index {
  build() {
    Button("XXX")
      .width(200)
      .height(100)
      .onTouch(() => {
        hilog.info(1001, 'Index', 'onTouch') // Avoid printing logs
      })
  }
}