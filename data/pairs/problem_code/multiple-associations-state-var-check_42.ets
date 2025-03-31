@Observed
class UIStyle {
  isChecked: boolean = false;
}

@Entry
@Component
struct Example4 {
  @State uiStyle: UIStyle = new UIStyle();

  build() {
    Column() {
      CompB({ subStyle: this.uiStyle })
      CompC({ subStyle: this.uiStyle })
      Text('Toggle Check')
        .onClick(() => {
          this.uiStyle.isChecked = !this.uiStyle.isChecked;
        })
    }
  }
}

@Component
struct CompB {
  @Link subStyle: UIStyle;

  build() {
    if (this.subStyle.isChecked) {
      Text('Component B Checked')
    } else {
      Text('Component B Unchecked')
    }
  }
}

@Component
struct CompC {
  @Link subStyle: UIStyle;

  build() {
    if (this.subStyle.isChecked) {
      Text('Component C Checked')
    } else {
      Text('Component C Unchecked')
    }
  }
}
