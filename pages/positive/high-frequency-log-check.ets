@Entry
@Component
struct Index {
  build() {
      Column() {
        Scroll()
          .onScroll(() => {
            const TAG = 'onScroll';
          })
      }
  }
}