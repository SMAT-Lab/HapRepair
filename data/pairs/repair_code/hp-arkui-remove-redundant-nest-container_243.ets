@Entry
@Component
struct NestedButtons {
  build() {
      Flex() {
        Button('Button A')
          .fontSize(16)
      }
      .justifyContent(FlexAlign.Center)
      .alignItems(FlexAlign.Center)
      .backgroundColor(Color.LightGray)
  }
}
