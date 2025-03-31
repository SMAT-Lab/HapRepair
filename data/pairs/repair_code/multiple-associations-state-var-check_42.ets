@Observed
class UIStyle {
  isChecked: boolean = false;
}

@Entry
@Component
struct Example4Fixed {
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
  @Link @Watch('onStyleChange') subStyle: UIStyle;
  @State localChecked: boolean = false;

  onStyleChange() {
    this.localChecked = this.subStyle.isChecked;
  }

  build() {
    if (this.localChecked) {
      Text('Component B Checked')
    } else {
      Text('Component B Unchecked')
    }
  }
}

@Component
struct CompC {
  @Link @Watch('onStyleChange') subStyle: UIStyle;
  @State localChecked: boolean = false;

  onStyleChange() {
    this.localChecked = this.subStyle.isChecked;
  }

  build() {
    if (this.localChecked) {
      Text('Component C Checked')
    } else {
      Text('Component C Unchecked')
    }
  }
}
