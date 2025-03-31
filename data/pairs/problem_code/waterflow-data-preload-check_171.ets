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
        .height(this.itemHeightArray[item % 60])
        .backgroundColor(this.colors[item % 3])
      }, (item: string) => item)
    }
    .cachedCount(2)
    .onReachEnd(() => {
      console.info("Load end")
      for (let i = 0; i < 70; i++) {
        this.dataSource.addBatchItem()
      }
    })
  }
}