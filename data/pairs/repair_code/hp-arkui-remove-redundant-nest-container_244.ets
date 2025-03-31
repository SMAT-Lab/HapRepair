@Entry
@Component
struct ComplexLayout {
  build() {
    Stack() {
      Rectangle()
        .width(100)
        .height(100)
        .backgroundColor(Color.Red)
        .padding(10)
        .borderRadius(10)
    }
    .shadow({ color: Color.Black, radius: 5, offset: { x: 0, y: 2 } })
  }
}
