@Entry
@Component
struct ColorComponent {
  @State red: number = 255;
  @State green: number = 255;
  @State blue: number = 255;

  updateRedColor() {
    animateTo({ curve: Curve.Linear, duration: 300 }, () => {
      this.red = (this.red === 255 ? 0 : 255);
    });
  }

  updateGreenColor() {
    animateTo({ curve: Curve.Linear, duration: 300 }, () => {
      this.green = (this.green === 255 ? 0 : 255);
    });
  }

  updateBlueColor() {
    animateTo({ curve: Curve.Linear, duration: 300 }, () => {
      this.blue = (this.blue === 255 ? 0 : 255);
    });
  }

  resetColors() {
    animateTo({ curve: Curve.Linear, duration: 300 }, () => {
      this.red = 255;
      this.green = 255;
      this.blue = 255;
    });
  }

  build() {
    Column() {
      Row()
        .backgroundColor(Color.rgb(this.red, this.green, this.blue))
        .height(200)
        .width(200)
      Button('Toggle Red')
        .onClick(() => {
          this.resetColors()
          this.updateRedColor()
        })
      Button('Toggle Green')
        .onClick(() => {
          this.resetColors()
          this.updateGreenColor()
        })
      Button('Toggle Blue')
        .onClick(() => {
          this.resetColors()
          this.updateBlueColor()
        })
      Button('Reset Colors')
        .onClick(() => this.resetColors())
    }
    .width('100%')
    .height('100%')
  }
}
