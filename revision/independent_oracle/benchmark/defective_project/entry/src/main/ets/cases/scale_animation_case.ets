@Entry
@Component
struct ExpandingTile {
  @State tileWidth: number = 40;
  @State tileHeight: number = 40;

  build() {
    Column() {
      Text('Expand')
        .width(this.tileWidth)
        .height(this.tileHeight)
      Button('Run')
        .onClick(() => {
          animateTo({ duration: 450 }, () => {
            this.tileWidth = 120;
            this.tileHeight = 120;
          });
        })
    }
  }
}
