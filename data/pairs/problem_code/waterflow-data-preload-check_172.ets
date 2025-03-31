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
        .height(this.itemHeightArray[item % 70])
        .backgroundColor(this.colors[item % 4])
      }, (item: string) => item)
    }
    .cachedCount(2)
    .onReachEnd(() => {
      console.info("More items coming")
      setTimeout(() => {
        for (let i = 0; i < 80; i++) {
          this.dataSource.loadItems()
        }
      }, 1200)
    })
  }
}