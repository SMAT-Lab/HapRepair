import hilog from '@ohos.hilog'
@Entry
@Component
struct ExampleComponent {
  @State message: string = '';

  build() {
    Column() {
      Button('点击更新状态并打印日志')
        .onClick(() => {
          this.message = 'Hello, World!';

          for (let i = 0; i < 10; i++) {
            for (let j=0; j<10; j++) {
              hilog.info(0x0000, 'TAG', '%{public}s', this.message);
            }
          }
        })
        .width('90%')
        .backgroundColor(Color.Green)
        .fontColor(Color.White)
        .margin({ top: 10 })
    }
    .justifyContent(FlexAlign.Start)
    .alignItems(HorizontalAlign.Center)
    .margin({ top: 15 })
  }
}