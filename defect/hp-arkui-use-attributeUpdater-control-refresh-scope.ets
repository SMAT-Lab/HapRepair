@Entry
@Component
struct MyColorComponent {
  @State selectedColor: string = "#FF0000";
  private colors: string[] = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00"];

  build() {
    Column() {
      Text('当前颜色: ' + this.selectedColor)
        .fontSize(20)
        .padding(10)

      List() {
        LazyForEach(this.colors, (color: string) => {
          ListItem() {
            ColorBox({ color: color })
              .onClick(() => {
                this.selectedColor = color;
              })
          }
        }, (color: string) => color)
      }
      .cachedCount(2)
      .width('100%')
      .height(200)
    }
  }
}

@Reusable
@Component
struct ColorBox {
  @State color: string = "#FFFFFF";  // 颜色状态变量

  aboutToReuse(params: Record<string, Object>): void {
    this.color = params.color as string;
  }

  build() {
      Text('颜色方块')
        .fontSize(16)
        .padding(5)
        .backgroundColor(this.color)
        .borderColor(this.color)
        .borderWidth(2)
        .borderStyle(BorderStyle.Solid)
        .width(50)
        .height(50)
  }
}
