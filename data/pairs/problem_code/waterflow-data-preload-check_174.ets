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
        .height(this.itemHeightArray[item % 25])
        .backgroundColor(this.colors[item % 6])
      }, (item: string) => item)
    }
    .cachedCount(2)
    .onReachEnd(() => {
      console.info("Scroll end detected")
      setTimeout(() => {
        for (let i = 0; i < 110; i++) {
          this.dataSource.expandList()
        }
      }, 1500)
    })
  }
}