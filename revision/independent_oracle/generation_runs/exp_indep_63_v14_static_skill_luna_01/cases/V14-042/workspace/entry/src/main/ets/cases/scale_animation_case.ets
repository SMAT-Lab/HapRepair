@Entry
@Component
struct ExpandingTile {
  @State tileWidth: number = 40;
  @State tileHeight: number = 40;

  build() {
    Column() {
      Text('Expand')
        .width(40)
        .height(40)
        .scale({ x: this.tileWidth / 40, y: this.tileHeight / 40, centerX: 0, centerY: 0 })
      Button('Run')
        .translate({ y: this.tileHeight - 40 })
        .onClick(() => {
          animateTo({ duration: 450 }, () => {
            this.tileWidth = 120;
            this.tileHeight = 120;
          });
        })
    }
  }
}
