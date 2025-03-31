import hilog from '@ohos.hilog';
@Entry
@Component
struct App {
  build() {
    List() {
      Text('Item')
    }.onItemDragMove(() => {
      hilog.debug(1001, 'App', 'onItemDragMove')
    })
  }
}