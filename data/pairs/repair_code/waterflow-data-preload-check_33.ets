@Entry
@Component
struct App {
  build() {

    LazyForEach(this.dataSource, (item: number) => {
      FlowItem() {
        ReusableFlowItem({ item: item })
      }
      .onAppear(() => {
        if (item + 20 == this.dataSource.totalCount()) {
          for (let i = 0; i < 100; i++) {
            this.dataSource.addLastItem()
          }
        }
      })
      .width('100%')
      .height(this.itemHeightArray[item % 100])
      .backgroundColor(this.colors[item % 5])
    }, (item: string) => item)
  }
}