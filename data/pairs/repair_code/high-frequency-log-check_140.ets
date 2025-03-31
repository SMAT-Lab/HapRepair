@Entry
@Component
struct App {
  build() {
    List() {
      Text('Item')
    }.onItemDragMove(() => {
      const TAG = 'onItemDragMove';
    })
    .width("100%")
    .height("100%")
  }
}