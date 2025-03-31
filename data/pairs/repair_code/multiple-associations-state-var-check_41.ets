@Observed
class UIStyle {
  fontColor: string = '#000000';
}

@Entry
@Component
struct Example3Fixed {
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
  @Link @Watch('onStyleChange') subStyle: UIStyle;
  @State localFontColor: string = '#000000';

  onStyleChange() {
    this.localFontColor = this.subStyle.fontColor;
  }

  build() {
    Text('Component A')
      .fontColor(this.localFontColor)
  }
}

@Component
struct CompC {
  @Link @Watch('onStyleChange') subStyle: UIStyle;
  @State localFontColor: string = '#000000';

  onStyleChange() {
    this.localFontColor = this.subStyle.fontColor;
  }

  build() {
    Text('Component C')
      .fontColor(this.localFontColor)
  }
}
