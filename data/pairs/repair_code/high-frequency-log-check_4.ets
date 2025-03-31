@Entry
@Component
struct Index {
  build() {
    Row() {
      Slider({ value: 0, min: 0, max: 100 })
        .enabled(false)
        .height(4)
        .width(100)
        .trackThickness(3)
        .blockColor(Color.Red)
        .blockSize({ width: 4, height: 4 })
        .onVisibleAreaChange([0.0, 1.0], (isVisible: boolean, currentRatio: number) => {
          const TAG = 'onVisibleAreaChange';
        })
    }
  }
}