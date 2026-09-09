@Entry
@Component
struct ExpandingTile {
  @State scaleFactor: number = 1;

  build() {
    Column() {
      Text('Expand')
        .width(40)
        .height(40)
        .scale({ x: this.scaleFactor, y: this.scaleFactor })
      Button('Run')
        .onClick(() => {
          animateTo({ duration: 450 }, () => {
            this.scaleFactor = 3;
          });
        })
    }
  }
}
