@Entry
@Component
struct SwipeHandle {
  private options: PanGestureOptions = new PanGestureOptions({ direction: PanDirection.Horizontal });

  build() {
    Column() {
      Text('Swipe target')
        .gesture(PanGesture(this.options))
      Button('Configure swipe')
        .onClick(() => {
          this.options.setDistance(5);
        })
    }
  }
}
