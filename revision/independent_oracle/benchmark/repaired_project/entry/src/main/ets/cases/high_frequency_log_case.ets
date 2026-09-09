@Entry
@Component
struct ActivityFeed {
  private scrollEventCount: number = 0;

  build() {
    Scroll() {
      Column() {
        Text('Activity')
      }
    }
    .onWillScroll(() => {
      this.scrollEventCount += 1;
    })
  }
}
