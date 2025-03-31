@Entry
@Component
struct App {
  build() {
    Row() {
      WaterFlow() {
        LazyForEach(this.sourceData, (item: number) => {
          FlowItem() {
            CustomFlowItem({ item: item })
          }
          .width('100%')
          .height(this.sizeArray[item % 50])
          .backgroundColor(this.hues[item % 2])
        }, (item: string) => item)
      }
      .cachedCount(2)
    }
  }
}