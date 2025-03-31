@Observed
class UIStyle {
  isChecked: boolean = false;
}

@Entry
@Component
struct Example2 {
  @State uiStyle: UIStyle = new UIStyle();

  build() {
    Column() {
      CompC({ subStyle: this.uiStyle })
      Text('Toggle Checked')
        .onClick(() => {
          this.uiStyle.isChecked = !this.uiStyle.isChecked;
        })
    }
  }
}

@Component
struct CompC {
  @Link subStyle: UIStyle;

  build() {
    if (this.subStyle.isChecked) {
      Text('Checked')
    } else {
      Text('Unchecked')
    }
  }
}
