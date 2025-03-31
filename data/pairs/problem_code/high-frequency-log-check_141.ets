import hilog from '@ohos.hilog';
@Entry
@Component
struct App {
  build() {
    Canvas(()=>{})
      .width('100%')
      .height('100%')
      .onAreaChange(() => {
        hilog.error(1001, 'App', 'onAreaChange')
    })
  }
}