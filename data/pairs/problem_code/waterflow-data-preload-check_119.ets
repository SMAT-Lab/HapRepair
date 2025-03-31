@Entry
@Component
struct App{
  build(){
    Row() {
      LazyForEach(this.itemSource, (item: number) => {
        FlowItem() {
          BaseFlowItem({ item: item })
        }
        .width('100%')
        .height(this.configHeight[item % 40])
        .backgroundColor(this.palette[item % 5])
      }, (item: string) => item)
    }
  }
}