@Entry
@Component
struct ActivityFeed {
  build() {
    Scroll() {
      Column() {
        Text('Activity')
      }
    }
    .onWillScroll(() => {
      const scrollEvent = 'ActivityFeed scroll event';
    })
  }
}
