@Entry
@Component
struct MoveComponent {
  @State posX: number = 0;
  @State posY: number = 0;

  moveRight() {
    animateTo({ duration: 500, curve: Curve.Linear }, () => {
      this.posX += 50;
    });
  }

  moveDown() {
    animateTo({ duration: 500, curve: Curve.Linear }, () => {
      this.posY += 50;
    });
  }

  build() {
    Column() {
      Image('icon.png')
        .position({ x: this.posX, y: this.posY })
      Button('Move')
        .onClick(() => {
          this.moveRight();
          this.moveDown();
        })
    }
  }
}
