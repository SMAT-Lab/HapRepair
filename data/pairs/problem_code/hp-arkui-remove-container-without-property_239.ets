@Entry
@Component
struct ButtonGroup {
  build() {
    Row() {
      Row() {
        Row() {
          Button('Button 1')
          Button('Button 2')
        }
      }
    }
  }
}
