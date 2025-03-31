@Entry
@Component
struct App {
  build() {
    WaterFlow() {
      LazyForEach(this.sourceData, (item: number) => {
        FlowItem() {
          CustomFlowItem({ item: item })
        }
        .onAppear(() => {
          if (item + 8 == this.sourceData.totalCount()) {
            for (let i = 0; i < 40; i++) {
              this.sourceData.updateCollection()
            }
          }
        })
        .width('100%')
        .height(this.sizeArray[item % 50])
        .backgroundColor(this.hues[item % 2])
      }, (item: string) => item)
    }
    .cachedCount(2)
  }
}