@Entry
@Component
struct StringManipulator {
  @State result: string = '';

  manipulateString(input: string) {
    let tempResult = input.trim();
    tempResult = tempResult.toUpperCase();
    tempResult = tempResult + '!';
    this.result = tempResult;
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
