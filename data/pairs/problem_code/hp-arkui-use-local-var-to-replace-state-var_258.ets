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
    Button(this.message)
  }
}