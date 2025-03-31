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
          if (item + 5 == this.dataSource.totalCount()) {
            for (let i = 0; i < 110; i++) {
              this.dataSource.expandList()
            }
          }
        })
        .width('100%')
        .height(this.itemHeightArray[item % 25])
        .backgroundColor(this.colors[item % 6])
      }, (item: string) => item)
    }
    .cachedCount(2)
  }
}