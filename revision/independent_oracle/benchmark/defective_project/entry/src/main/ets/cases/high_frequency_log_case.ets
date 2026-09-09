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
      console.info('ActivityFeed scroll event');
    })
  }
}
