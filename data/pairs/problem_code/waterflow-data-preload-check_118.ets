@Entry
@Component
struct App {
  build() {
    Row() {
      WaterFlow() {
        LazyForEach(this.elements, (item: number) => {
          FlowItem() {
            ComponentFlow({ item: item })
          }
          .width('100%')
          .height(this.heightMap[item % 35])
          .backgroundColor(this.colorMap[item % 4])
        }, (item: string) => item)
      }
      .cachedCount(2)
    }
  }
}