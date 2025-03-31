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
          if (item + 10 == this.dataSource.totalCount()) {
            for (let i = 0; i < 20; i++) {
              this.dataSource.addMoreItems()
            }
          }
        })
        .width('100%')
        .height(this.itemHeightArray[item % 20])
        .backgroundColor(this.colors[item % 4])
      }, (item: string) => item)
    }
    .cachedCount(2)
  }
}