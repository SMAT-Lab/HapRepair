@Entry
@Component
struct Index {
  build() {
    Button("")
      .width(200)
      .height(100)
      .onMouse(() => {
        const TAG = 'onMouse';
      })
  }
}