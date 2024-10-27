@Entry
@Component
struct MyComponent {
  @State currentStatus: string = "initial";

  updateStatus(newStatus: string): void {
    this.currentStatus = newStatus;
  }

  build() {
    Stack() {
      Text("这是一个按钮")
        .fontSize(16)
        .textAlign(TextAlign.Center)
    }
    .backgroundColor("blue")
    .width(200)
    .height(100)
  }
}