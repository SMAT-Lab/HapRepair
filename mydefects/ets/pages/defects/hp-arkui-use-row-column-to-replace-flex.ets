@Entry
@Component
struct MyComponent {
  build() {
    Flex({ direction: FlexDirection.Column }) {
      Text('Replace Flex with Column/Row')
        .fontSize(14)
        .height(30)
        .margin({
          top: 10,
          bottom: 10
        })

      Flex({ direction: FlexDirection.Column }) {
        Image('resources/image1.jpg')
          .width(300)
          .height(150)
          .backgroundColor(Color.Blue)
        Image('resources/image2.jpg')
          .width(300)
          .height(150)
          .backgroundColor(Color.Green)
        Image('resources/image3.jpg')
          .width(300)
          .height(150)
          .backgroundColor(Color.Red)
      }.width('100%').padding(10)

      Flex({ direction: FlexDirection.Row }) {
        Text('Item 1').padding(10)
        Text('Item 2').padding(10)
        Button('Submit').padding(10)
      }.width('100%').padding(10)
    }.padding(10)
  }
}
