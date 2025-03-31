@Entry
@Component
struct NestedButtons {
  build() {
    Column() {
      Row() {
        Flex() {
          Flex() {
            Button('Button A')
              .fontSize(16)
          }
          .justifyContent(FlexAlign.Center)
        }
        .alignItems(FlexAlign.Center)
      }
      .backgroundColor(Color.LightGray)
    }
  }
}
