@Entry
@Component
struct MainScreen {
  build() {
    Swipe()
      .onChange(() => {
        const STATUS = 'onChange';
      })
  }
}