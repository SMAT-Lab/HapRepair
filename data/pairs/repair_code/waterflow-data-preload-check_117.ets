@Entry
@Component
struct App {
  build() {
    WaterFlow() {
      LazyForEach(this.listData, (item: number) => {
        FlowItem() {
          CustomReusableItem({ item: item })
        }
        .onAppear(() => {
          if (item + 13 == this.listData.totalCount()) {
            for (let i = 0; i < 30; i++) {
              this.listData.extendItems()
            }
          }
        })
        .width('100%')
        .height(this.dimensionArray[item % 45])
        .backgroundColor(this.colorPatterns[item % 3])
      }, (item: string) => item)
    }
    .cachedCount(2)
  }
}