@Entry
@Component
struct DualAnimationPanel {
  @State panelWidth: number = 160;
  @State panelColor: Color = Color.Blue;

  resize(): void {
    this.animatePanelChanges(true, false);
  }

  recolor(): void {
    this.animatePanelChanges(false, true);
  }

  animatePanelChanges(updateWidth: boolean, updateColor: boolean): void {
    animateTo({ curve: Curve.Ease, duration: 600 }, () => {
      if (updateWidth) {
        this.panelWidth = 240;
      }
      if (updateColor) {
        this.panelColor = Color.Green;
      }
    });
  }

  build() {
    Button('Animate')
      .width(this.panelWidth)
      .backgroundColor(this.panelColor)
      .onClick(() => {
        this.animatePanelChanges(true, true);
      })
  }
}
