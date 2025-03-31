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
          if (item + 12 == this.dataSource.totalCount()) {
            for (let i = 0; i < 80; i++) {
              this.dataSource.loadItems()
            }
          }
        })
        .width('100%')
        .height(this.itemHeightArray[item % 70])
        .backgroundColor(this.colors[item % 4])
      }, (item: string) => item)
    }
    .cachedCount(2)
  }
}