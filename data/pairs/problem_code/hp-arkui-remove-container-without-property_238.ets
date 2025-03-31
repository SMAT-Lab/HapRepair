@Entry
@Component
struct ListExample {
  build() {
    List() {
      ForEach([1, 2, 3], (item) => {
        ListItem() {
          Stack() {
            Stack() {
              Text('Item ' + item)
            }
          }
        }
      },(item:number)=>item.toString())
    }
    .cachedCount(2)
    .width("100%")
    .height('100%')
  }
}
