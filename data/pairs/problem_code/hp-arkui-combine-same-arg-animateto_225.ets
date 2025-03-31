@Entry
@Component
struct SizeChanger {
  @State width: number = 100;
  @State height: number = 100;

  increaseWidth() {
    animateTo({ duration: 800, curve: Curve.EaseInOut }, () => {
      this.width += 50;
    });
  }

  increaseHeight() {
    animateTo({ duration: 800, curve: Curve.EaseInOut }, () => {
      this.height += 50;
    });
  }

  build() {
    Column() {
      Rectangle()
        .width(this.width)
        .height(this.height)
        .backgroundColor(Color.Blue)
      Button('Increase Size')
        .onClick(() => {
          this.increaseWidth();
          this.increaseHeight();
        })
    }
  }
}
