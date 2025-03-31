@Entry
@Component
struct App {
  build() {
    Row() {
      LazyForEach(this.data, item => {
        GridItem() {
          Column() {
            Image(this.productData[item % 14].uri)
              .width('100%')
              .height('40%')
              .objectFit(ImageFit.Contain)
              .margin({ bottom: 40 })
            Text(this.productData[item % 14].title)
              .fontSize(16 * this.ratio)
              .fontWeight(600)
              .margin({ bottom: 10, left: 10 })
            Row() {
              Text(this.productData[item % 14].labels)
                .fontSize(10 * this.ratio)
                .border({ width: 1, color: '#FA808080' })
                .margin({ bottom: 2, left: 10 })
                .padding(2)
            }
            .margin({ bottom: 2 })

            Text(this.labels)
              .fontSize(16 * this.ratio)
              .fontColor(Color.Red)
              .margin({ left: 10 })
          }
          .alignItems(HorizontalAlign.Start)
        }
        .width('95%')
        .height(300)
        .border({ width: 1, color: '#70808080', radius: 10 })
        .margin({ top: 3, bottom: 3 })
        .backgroundColor(Color.White)
      }, item => item.toString())
    }
  }
}