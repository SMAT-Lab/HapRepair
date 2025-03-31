@Entry
@Component
struct Dashboard {
  build() {
    Scroll()
      .onScrollStart(() => {
        const LABEL = 'onScroll';
      })
  }
}