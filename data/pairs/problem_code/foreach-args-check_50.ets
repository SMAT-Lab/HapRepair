@Entry
@Component
struct App {
  build() {
    Column() {
      ForEach(this.list, (item) => {
        Text(item.toString())
          .width('90%')
          .height(72)
          .backgroundColor('#fff')
          .borderRadius(15)
          .fontSize(24)
          .textAlign(TextAlign.Center)
          .margin(10)
      }, item => item)
    }.width('100%').backgroundColor('#F1F3F5').padding({ bottom: 10 })
  }
}
