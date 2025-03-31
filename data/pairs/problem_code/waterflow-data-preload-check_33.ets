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
        .height(this.itemHeightArray[item % 100])
        .backgroundColor(this.colors[item % 5])
      }, (item: string) => item)
    }
    .cachedCount(2)
    .onReachEnd(() => {
      console.info("onReachEnd")
      setTimeout(() => {
        for (let i = 0; i < 100; i++) {
          this.datasource.AddLastItem()
        }
      }, 1000)
    })
  }
}