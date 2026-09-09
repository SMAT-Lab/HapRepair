@Entry
@Component
struct DualAnimationPanel {
  @State panelWidth: number = 160;
  @State panelColor: Color = Color.Blue;

  resize(): void {
    this.panelWidth = 240;
  }

  recolor(): void {
    this.panelColor = Color.Green;
  }

  animatePanelChanges(): void {
    animateTo({ curve: Curve.Ease, duration: 600 }, () => {
      this.resize();
      this.recolor();
    });
  }

  build() {
    Button('Animate')
      .width(this.panelWidth)
      .backgroundColor(this.panelColor)
      .onClick(() => {
        this.animatePanelChanges();
      })
  }
}
