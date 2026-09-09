@Entry
@Component
struct HighRateAnimation {
  @State offset: number = 0;

  build() {
    Button('Move')
      .translate({ x: this.offset })
      .onClick(() => {
        animateTo({
          duration: 800,
          expectedFrameRateRange: {
            expected: 120,
            min: 120,
            max: 120
          }
        }, () => {
          this.offset = 100;
        });
      })
  }
}
