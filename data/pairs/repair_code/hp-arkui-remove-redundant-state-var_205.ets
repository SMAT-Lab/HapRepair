@Entry
@Component
struct MyComponent {

  appendMsg(newMsg: String) : string {
    this.message += newMsg;
    return this.message;
  }

  build() {
    Column() {
      Stack() {
      }
      .backgroundColor("black")
      .width(200)
      .height(400)
      Button("move")
    }
  }
}