@Entry
@Component
struct App {
  build() {
    Column({ space: 2 }) {
      WaterFlow() {
        LazyForEach(this.dataSource, (item: number) => {
          FlowItem() {
            ReusableFlowItem({ item: item })
          }
          .width('100%')
          .height(this.itemHeightArray[item % 20])
          .backgroundColor(this.colors[item % 2])
        }, (item: string) => item)
      }
      .cachedCount(2)
      .onReachEnd(() => {
        console.info("Adding more data")
        setTimeout(() => {
          for (let i = 0; i < 50; i++) {
            this.dataSource.addLastItem()
          }
        }, 800)
      })
      .columnsTemplate("1fr 1fr")
      .columnsGap(10)
      .rowsGap(5)
      .backgroundColor(0xFAEEE0)
      .width('100%')
      .height('100%')
    }
  }
}