@Entry
@Component
struct ExampleComponent {
  build() {
    Column() {
      // 第一层无属性的 Column
      Column() {
        // 第二层无属性的 Column
        Column() {
          Text('Hello World')
        }
      }
    }
  }
}
