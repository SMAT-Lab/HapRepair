@Observed
class UIStyle {
  fontSize: number = 0;
}

@Entry
@Component
struct Example1Fixed {
  @State uiStyle: UIStyle = new UIStyle();

  build() {
    Column() {
      CompA({ subStyle: this.uiStyle })
      CompB({ subStyle: this.uiStyle })
      Text('Change Font Size')
        .onClick(() => {
          this.uiStyle.fontSize = 24;
        })
    }
  }
}

@Component
struct CompA {
  @Link @Watch('onStyleChange') subStyle: UIStyle;
  @State localFontSize: number = 0;

  onStyleChange() {
    this.localFontSize = this.subStyle.fontSize;
  }

  build() {
    Text('Component A')
      .fontSize(this.localFontSize)
  }
}

@Component
struct CompB {
  @Link @Watch('onStyleChange') subStyle: UIStyle;
  @State localFontSize: number = 0;

  onStyleChange() {
    this.localFontSize = this.subStyle.fontSize;
  }

  build() {
    Text('Component B')
      .fontSize(this.localFontSize)
  }
}
