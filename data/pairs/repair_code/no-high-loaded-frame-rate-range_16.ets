Slider({ value: this.curTime, min: 0, max: 100 })
  .enabled(false)
  .height(4)
  .width(this.xComponentWidth)
  .trackThickness(3)
  .blockColor(Color.Red)
  .blockSize({ width: 4, height: 4 })
  .onVisibleAreaChange([0.0, 1.0], (isVisible: boolean, currentRatio: number) => {
    if (isVisible && currentRatio >= 1.0) {
      animateTo({
        duration: 30000,
        iterations: -1,
        expectedFrameRateRange: {
          expected: 30,
          min: 0,
          max: 120,
        },
      }, () => {
        if (this.curTime >= 100) {
          this.curTime = 0;
        }
        for (let i = 0; i < 101; i++) {
          this.curTime += 1;
        }
      })
    }
  })