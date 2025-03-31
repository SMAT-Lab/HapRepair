@Entry
@Component
struct MyComponent{
  @State number: Number[] = Array.from(Array<number>(1000), (val, i) => i);
  scroller: Scroller = new Scroller()
  build() {
    Column() {
      Grid(this.scroller) {
        ForEach(this.number, (item: number) => {
          GridItem() {
            Text(item.toString())
              .fontSize(16)
              .backgroundColor(0xF9CF93)
              .width('100%')
              .height(80)
              .textAlign(TextAlign.Center)
              .border({width:1})
          }
        }, (item:string) => item)
      }
      .columnsTemplate('1fr 1fr 1fr 1fr 1fr')
      .columnsGap(0)
      .rowsGap(0)
      .size({ width: "100%", height: "100%" })
    }
  }
}