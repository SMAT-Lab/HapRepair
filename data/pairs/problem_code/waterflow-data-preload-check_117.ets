@Entry
@Component
struct App {
  build() {
    Row() {
      WaterFlow() {
        LazyForEach(this.listData, (item: number) => {
          FlowItem() {
            CustomReusableItem({ item: item })
          }
          .width('100%')
          .height(this.dimensionArray[item % 45])
          .backgroundColor(this.colorPatterns[item % 3])
        }, (item: string) => item)
      }
      .cachedCount(2)
    }
  }
}