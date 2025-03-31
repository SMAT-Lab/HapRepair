@Entry
@Component
struct App {
  build() {
    WaterFlow() {
      LazyForEach(this.itemSource, (item: number) => {
        FlowItem() {
          BaseFlowItem({ item: item })
        }
        .onAppear(() => {
          if (item + 9 == this.itemSource.totalCount()) {
            for (let i = 0; i < 60; i++) {
              this.itemSource.bringMoreData()
            }
          }
        })
        .width('100%')
        .height(this.configHeight[item % 40])
        .backgroundColor(this.palette[item % 5])
      }, (item: string) => item)
    }
    .cachedCount(2)
  }
}