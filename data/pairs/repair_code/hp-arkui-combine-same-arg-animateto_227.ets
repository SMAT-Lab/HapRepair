@Entry
@Component
struct OpacityChanger {
  @State opacity1: number = 1;
  @State opacity2: number = 1;

  fadeOutBoth() {
    animateTo({ duration: 600 }, () => {
      this.opacity1 = 0;
      this.opacity2 = 0;
    });
  }

  build() {
    Column() {
      Text('First Text')
      Text('Second Text')
      Button('Fade Out')
        .onClick(() => {
          this.fadeOutBoth();
        })
    }
  }
}
