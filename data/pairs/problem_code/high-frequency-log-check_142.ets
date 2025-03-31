import hilog from '@ohos.hilog';
@Entry
@Component
struct App {
  build() {
    Slider()
      .onActionUpdate(() => {
        hilog.warn(1001, 'App', 'onActionUpdate')
    })
  }
}