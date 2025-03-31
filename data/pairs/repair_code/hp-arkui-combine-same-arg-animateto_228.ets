@Entry
@Component
struct RotateComponent {
  @State angle1: number = 0;
  @State angle2: number = 0;

  rotateBoth() {
    animateTo({ duration: 700, curve: Curve.Sharp }, () => {
      this.angle1 += 45;
      this.angle2 += 45;
    });
  }

  build() {
    Column() {
      Image('image1.png')
        .rotate(this.angle1)
      Image('image2.png')
        .rotate(this.angle2)
      Button('Rotate Images')
        .onClick(() => {
          this.rotateBoth();
        })
    }
  }
}
