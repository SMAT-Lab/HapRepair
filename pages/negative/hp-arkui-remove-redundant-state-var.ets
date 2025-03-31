// 建议移除不关联UI组件的状态变量设置。
//
// 通用丢帧场景下，建议优先修改。

@Entry
@Component
struct MyComponent {
  @State message: string = "";
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