@Observed
class UIStyle {
  isChecked: boolean = false;
}

@Entry
@Component
struct Example2Fixed {
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
  @Link @Watch('onStyleChange') subStyle: UIStyle;
  @State localChecked: boolean = false;

  onStyleChange() {
    this.localChecked = this.subStyle.isChecked;
  }

  build() {
    if (this.localChecked) {
      Text('Checked')
    } else {
      Text('Unchecked')
    }
  }
}
