@Entry
@Component
struct App {
  build() {
    WaterFlow() {
      LazyForEach(this.dataSource, (item: number) => {
        FlowItem() {
          ReusableFlowItem({ item: item })
        }
        .width('100%')
        .height(this.itemHeightArray[item % 80])
        .backgroundColor(this.colors[item % 2])
      }, (item: string) => item)
    }
    .cachedCount(2)
    .onReachEnd(() => {
      console.info("End reached")
      setTimeout(() => {
        for (let i = 0; i < 40; i++) {
          this.dataSource.addItems()
        }
      }, 1000)
    })
  }
}