@Entry
@Component
struct MyComponent {
  @State count: number = 0;   // 状态变量

  incrementCount(): void {
    this.count += 1;  // 状态变量被更新
  }

  resetCount(): void {
    this.count = 0;  // 状态变量被更新
  }

  build() {
    Stack() {
      Text("Current count")
        .fontSize(16)
        .textAlign(TextAlign.Center)

      Button("Increment")
        .fontSize(16)
        .backgroundColor(Color.Gray)
        .padding(10)

      Button("Reset")
        .fontSize(16)
        .backgroundColor(Color.Red)
        .padding(10)
    }
    .width('100%')
    .height('100%')
  }
}
