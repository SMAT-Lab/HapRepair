@Entry
@Component
struct MyComponent {
  // 定义状态变量
  @State text: string = "Initial Text";
  @State color: Color = Color.Blue;
  @State button_msg: string = "Change Text and Color"

  build() {
    Column() {
      // 使用状态变量来设置文本内容和颜色
      Text(this.text)
        .fontSize(20)
        .fontColor(this.color)
        .padding(20)

      Button(this.button_msg)
        .onClick(() => {
          this.text = "Hello";
        })
        .fontSize(16)
        .backgroundColor(Color.Gray)
        .padding(10)
    }
  }
}