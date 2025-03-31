@Entry
@Component
struct App{
  build() {
    Row() {
      LazyForEach(this.dataSource, (item: number) => {
        FlowItem() {
          ReusableFlowItem({ item: item })
        }
        .width('100%')
        .height(this.itemHeightArray[item % 30])
        .backgroundColor(this.colors[item % 3])
      }, (item: string) => item)
    }
  }
}