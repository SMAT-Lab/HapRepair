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
            expected: 60,
            min: 30,
            max: 60
          }
        }, () => {
          this.offset = 100;
        });
      })
  }
}
