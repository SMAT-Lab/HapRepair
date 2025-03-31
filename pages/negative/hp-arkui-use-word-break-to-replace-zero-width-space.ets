// 建议使用word-break替换零宽空格(\u200b)。
//
// 根据ArkUI编程规范，建议修改。

@Component
export struct MyComponent {
  private diskName: string = '';

  build() {
    Text(this.diskName.split("").join("\u200B"))
      .textAlign(TextAlign.Start)
  }
}