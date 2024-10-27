@Entry
@Component
struct MyComponent {
  @State boxWidth: number = 50;
  @State boxHeight: number = 50;
  build() {
    Column() {
      Text()
        .backgroundColor(Color.Blue)
        .width(this.boxWidth)
        .height(this.boxHeight)
      Button('调整大小')
        .backgroundColor(Color.Green)
        .fontColor(Color.White)
        .fontSize(20)
        .padding(10)
        .margin({ top: 30 })
        .onClick(() => {
          animateTo({ duration: 1000 }, () => {
            this.boxWidth = 200;
            this.boxHeight = 200;
          })
        })
    }
  }
}