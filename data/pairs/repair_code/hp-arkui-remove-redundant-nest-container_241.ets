@Entry
@Component
struct CardList {
  build() {
    List() {
      ForEach([1, 2, 3], (item) => {
        ListItem() {
          Column() {
            Text('Card ' + item)
              .padding(10)
          }
          .backgroundColor(Color.White)
          .borderRadius(8)
          .margin(5)
        }
      }, (item: number)=> item.toString())
    }
    .height('100%')
    .width('100%')
    .cachedCount(2)
  }
}
