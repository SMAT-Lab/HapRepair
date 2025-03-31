@Entry
@Component
struct StringManipulator {
  @State result: string = '';

  manipulateString(input: string) {
    this.result = input.trim();
    this.result = this.result.toUpperCase();
    this.result = this.result + '!';
  }

  build() {
    Column() {
      Text('结果：' + this.result)
        .fontSize(16)
      Button('处理字符串')
        .onClick(() => this.manipulateString(' hello world '))
    }
  }
}
