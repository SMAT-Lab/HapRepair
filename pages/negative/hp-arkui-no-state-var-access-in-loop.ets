// 避免在for、while等循环逻辑中频繁读取状态变量。
//
// 通用丢帧场景下，建议优先修改。

import hilog from '@ohos.hilog'
@Entry
@Component
struct MyComponent{
  @State message: string = '';
  build() {
    Column() {
      Button('点击打印日志')
        .onClick(() => {
          this.message = 'click';
          for (let i = 0; i < 10; i++) {
            hilog.info(0x0000, 'TAG', '%{public}s', this.message);
          }
        })
        .width('90%')
        .backgroundColor(Color.Blue)
        .fontColor(Color.White)
        .margin({
          top: 10
        })
    }
    .justifyContent(FlexAlign.Start)
    .alignItems(HorizontalAlign.Center)
    .margin({
      top: 15
    })
  }
}