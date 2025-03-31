@Entry
@Component
struct App {
  Row() {
    Text('1')
      .fontWeight(FontWeight.Bold)
      .backgroundColor('black')
      .fontColor('white')
      .fontSize(200)
      .width(200)
      .height(200)
      .lineHeight(200)
      .textAlign(TextAlign.Center)

      .onTouch((event: TouchEvent) => {
        const text = JSON.stringify(event.touches)
      })
  }
}
