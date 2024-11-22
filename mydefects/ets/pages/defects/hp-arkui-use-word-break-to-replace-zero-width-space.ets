@Entry
@Component
struct MyComponent {
  private longText: string = 'ThisIsAVeryLongTextWithoutSpaces';

  build() {
    // 使用零宽空格(\u200b)来强制文本换行
    Text(this.longText.split("").join("\u200B"))
      .fontSize(16)
      .padding(10)
      .textAlign(TextAlign.Start)
  }
}