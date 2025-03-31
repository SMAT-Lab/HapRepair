@Entry
@Component
struct Index {
  build() {
    Button("XXX")
      .width(200)
      .height(100)
      .onTouch(() => {
        const TAG = 'onTouch';
      })
  }
}