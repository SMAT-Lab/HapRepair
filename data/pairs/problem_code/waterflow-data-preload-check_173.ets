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
        .height(this.itemHeightArray[item % 90])
        .backgroundColor(this.colors[item % 5])
      }, (item: string) => item)
    }
    .cachedCount(2)
    .onReachEnd(() => {
      console.info("Appending items")
      for (let i = 0; i < 90; i++) {
        this.dataSource.increaseCollection()
      }
    })
  }
}