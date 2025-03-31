@Observed
class UIStyle {
  fontSize: number = 0;
}

@Entry
@Component
struct Example1 {
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
  @Link subStyle: UIStyle;

  build() {
    Text('Component A')
      .fontSize(this.subStyle.fontSize)
  }
}

@Component
struct CompB {
  @Link subStyle: UIStyle;

  build() {
    Text('Component B')
      .fontSize(this.subStyle.fontSize)
  }
}
