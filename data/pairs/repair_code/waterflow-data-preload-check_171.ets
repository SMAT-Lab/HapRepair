@Entry
@Component
struct App {
  build() {
    WaterFlow() {
      LazyForEach(this.dataSource, (item: number) => {
        FlowItem() {
          ReusableFlowItem({ item: item })
        }
        .onAppear(() => {
          if (item + 15 == this.dataSource.totalCount()) {
            for (let i = 0; i < 70; i++) {
              this.dataSource.addBatchItem()
            }
          }
        })
        .width('100%')
        .height(this.itemHeightArray[item % 60])
        .backgroundColor(this.colors[item % 3])
      }, (item: string) => item)
    }
    .cachedCount(2)
  }
}