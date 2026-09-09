@Entry
@Component
struct DualAnimationPanel {
  @State panelWidth: number = 160;
  @State panelColor: Color = Color.Blue;

  resize(): void {
    animateTo({ curve: Curve.Ease, duration: 600 }, () => {
      this.panelWidth = 240;
    });
  }

  recolor(): void {
    animateTo({ curve: Curve.Ease, duration: 600 }, () => {
      this.panelColor = Color.Green;
    });
  }

  build() {
    Button('Animate')
      .width(this.panelWidth)
      .backgroundColor(this.panelColor)
      .onClick(() => {
        this.resize();
        this.recolor();
      })
  }
}
