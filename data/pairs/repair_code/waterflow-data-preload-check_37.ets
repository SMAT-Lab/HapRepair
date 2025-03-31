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
          if (item + 25 == this.dataSource.totalCount()) {
            for (let i = 0; i < 40; i++) {
              this.dataSource.addItems()
            }
          }
        })
        .width('100%')
        .height(this.itemHeightArray[item % 80])
        .backgroundColor(this.colors[item % 2])
      }, (item: string) => item)
    }
    .cachedCount(2)
  }
}
