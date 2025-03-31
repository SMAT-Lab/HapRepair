@Entry
@Component
struct CardList {
  build() {
    List() {
      ForEach([1, 2, 3], (item) => {
        ListItem() {
          Column() {
            Column() {
              Column() {
                Text('Card ' + item)
              }
              .padding(10)
            }
            .backgroundColor(Color.White)
          }
          .borderRadius(8)
          .margin(5)
        }
      }, (item:number)=>item.toString())
    }
    .cachedCount(2)
    .width("100%")
    .height('100%')
  }
}
