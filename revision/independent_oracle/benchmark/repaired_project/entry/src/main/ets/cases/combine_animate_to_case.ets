@Entry
@Component
struct DualAnimationPanel {
  @State panelWidth: number = 160;
  @State panelColor: Color = Color.Blue;

  animatePanel(): void {
    animateTo({ curve: Curve.Ease, duration: 600 }, () => {
      this.panelWidth = 240;
      this.panelColor = Color.Green;
    });
  }

  build() {
    Button('Animate')
      .width(this.panelWidth)
      .backgroundColor(this.panelColor)
      .onClick(() => {
        this.animatePanel();
      })
  }
}
