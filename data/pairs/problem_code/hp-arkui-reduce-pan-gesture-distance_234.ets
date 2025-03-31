@Entry
@Component
struct DraggableBox {
  @State offsetX: number = 0;
  @State offsetY: number = 0;
  private panOption: PanGestureOptions = new PanGestureOptions({ direction: PanDirection.All });

  build() {
    Column() {
      Rectangle()
        .width(100)
        .height(100)
        .backgroundColor(Color.Blue)
        .translate({ x: this.offsetX, y: this.offsetY })
        .gesture(
          PanGesture(this.panOption)
            .onActionStart(() => {
              console.info('Pan started');
            })
            .onActionUpdate((event: GestureEvent) => {
              if (event) {
                this.offsetX += event.offsetX;
                this.offsetY += event.offsetY;
              }
            })
            .onActionEnd(() => {
              console.info('Pan ended');
            })
        )

      Button('设置拖动距离')
        .onClick(() => {
          // 设定的距离超过阈值10
          this.panOption.setDistance(50);
        })
    }
  }
}
