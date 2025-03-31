
  func1() {
    animateTo({ curve: Curve.Sharp, duration: 1000 }, () => {
      this.textWidth = (this.textWidth === 100 ? 200 : 100);
    });
  }

  func2() {
    animateTo({ curve: Curve.Sharp, duration: 1000 }, () => {
      this.color = (this.color === Color.Yellow ? Color.Red : Color.Yellow);
    });
  }

  build() {
    Column() {
      Row()
        .width(this.textWidth)
        .height(10)
        .backgroundColor(this.color)
      Text('click')
        .onClick(() => {
          this.func1();
          this.func2();
        })