@Entry
@Component
struct MyComponent {
  @State message: string = '';
  appendMsg(newMsg: String) {
      let message = this.message;
      message += newMsg;
      message += ";";
      message += "<br/>";
      this.message = message;
  }
  build() {
    Text(this.message)
  }
}