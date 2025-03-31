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
          if (item + 20 == this.dataSource.totalCount()) {
            for (let i = 0; i < 50; i++) {
              this.dataSource.addNewItem()
            }
          }
        })
        .width('100%')
        .height(this.itemHeightArray[item % 100])
        .backgroundColor(this.colors[item % 5])
      }, (item: string) => item)
    }
    .cachedCount(2)
  }
}
