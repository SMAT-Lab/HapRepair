@Entry
@Component
struct Index {
  build() {
    Button()
      .onClick(() => {
        animateTo({
          duration: 1200,
          iterations: 10,
          expectedFrameRateRange: { 
            expected: 30,
            min: 0,
            max: 120,
          },
        }, () => {
        })
      })
  }
}