@Entry
@Component
struct ExampleComponent {
  @State boxSize: number = 100;
  @State boxOpacity: number = 1;

  resizeBox() {
    animateTo({ curve: Curve.EaseInOut, duration: 500 }, () => {
      this.boxSize = (this.boxSize === 100 ? 200 : 100);
    });
  }

  changeOpacity() {
    animateTo({ curve: Curve.EaseInOut, duration: 500 }, () => {
      this.boxOpacity = (this.boxOpacity === 1 ? 0.5 : 1);
    });
  }

  build() {
    Column() {
      Row()
        .width(this.boxSize)
        .height(this.boxSize)
        .opacity(this.boxOpacity)
        .backgroundColor(Color.Blue)
      Button('Animate')
        .onClick(() => {
          this.resizeBox();
          this.changeOpacity();
        })
    }
    .width('100%')
    .height('100%')
  }
}
