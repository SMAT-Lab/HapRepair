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
            expected: 120,
            min: 120,
            max: 120,
          },
        }, () => {
        })
      })
  }
}