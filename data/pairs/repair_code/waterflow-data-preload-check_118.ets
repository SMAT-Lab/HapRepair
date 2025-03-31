@Entry
@Component
struct App {
  build() {
    WaterFlow() {
      LazyForEach(this.elements, (item: number) => {
        FlowItem() {
          ComponentFlow({ item: item })
        }
        .onAppear(() => {
          if (item + 3 == this.elements.totalCount()) {
            for (let i = 0; i < 20; i++) {
              this.elements.addMore()
            }
          }
        })
        .width('100%')
        .height(this.heightMap[item % 35])
        .backgroundColor(this.colorMap[item % 4])
      }, (item: string) => item)
    }
    .cachedCount(2)
  }
}