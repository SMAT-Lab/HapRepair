@Entry
@Component
struct ListExample {
  build() {
    List() {
      ForEach([1, 2, 3], (item) => {
        ListItem() {
          Text('Item ' + item)
        }
      }, (item:number) => item.toString())
    }
    .height('100%')
    .width('100%')
    .cachedCount(2)
  }
}
