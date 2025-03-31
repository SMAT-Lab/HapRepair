import hilog from '@ohos.hilog';
@Entry
@Component
struct Home {
  build() {
    TextInput()
      .onTextChange(() => {
        hilog.debug(5001, 'Home', 'onTextChange')
      })
  }
}