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
          if (item + 18 == this.dataSource.totalCount()) {
            for (let i = 0; i < 60; i++) {
              this.dataSource.fetchNewData()
            }
          }
        })
        .width('100%')
        .height(this.itemHeightArray[item % 30])
        .backgroundColor(this.colors[item % 3])
      }, (item: string) => item)
    }
    .cachedCount(2)
  }
}