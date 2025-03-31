import hilog from '@ohos.hilog';
@Entry
@Component
struct Index {
  build() {
    Scroll(this.scroller) {
      Column() {
        Text(this.message)
        .fontSize(50)
        .fontWeight(FontWeight.Bold)
      }.onScroll((xOffset: number, yOffset: number) => {
        console.info(xOffset + ' ' + yOffset)
      })
    }
  }
}
