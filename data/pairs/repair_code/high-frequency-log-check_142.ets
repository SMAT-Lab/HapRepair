@Entry
@Component
struct App {
  build() {
    Slider()
      .onActionUpdate(() => {
        const TAG = 'onActionUpdate';
    })
  }
}