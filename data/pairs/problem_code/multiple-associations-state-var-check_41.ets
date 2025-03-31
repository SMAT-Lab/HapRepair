@Observed
class UIStyle {
  fontColor: string = '#000000';
}

@Entry
@Component
struct Example3 {
  @State uiStyle: UIStyle = new UIStyle();

  build() {
    Column() {
      CompA({ subStyle: this.uiStyle })
      CompC({ subStyle: this.uiStyle })
      Text('Change Font Color')
        .onClick(() => {
          this.uiStyle.fontColor = '#FF0000';
        })
    }
  }
}

@Component
struct CompA {
  @Link subStyle: UIStyle;

  build() {
    Text('Component A')
      .fontColor(this.subStyle.fontColor)
  }
}

@Component
struct CompC {
  @Link subStyle: UIStyle;

  build() {
    Text('Component C')
      .fontColor(this.subStyle.fontColor)
  }
}
