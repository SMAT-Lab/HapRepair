@Entry
@Component
struct LayeredLabel {
  build() {
    Stack() {
      Text('Layered')
        .width('100%')
        .backgroundColor(Color.Yellow)
    }
    .backgroundColor(Color.Pink)
    .height(100)
  }
}
