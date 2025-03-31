// 建议使用临时变量替换状态变量。
//
// 通用丢帧场景下，建议优先修改。

@Entry
@Component
struct MyComponent {
  @State message: string = '';
  appendMsg(newMsg: String) {
    this.message += newMsg;
    this.message += ";";
    this.message += "<br/>";
  }
  build() {
    // 业务代码...
  }
}