import hilog from '@ohos.hilog';

@Entry
@Component
struct DragTest {
  build() {
    Column() {
      List() {
        ListItem() {
          Text('Item 1');
        }
        ListItem() {
          Text('Item 2');
        }
        .onItemDragMove(() => {
          hilog.info(1002, 'DragTest', 'onItemDragMove')
        })
      }
      .width('100%')
      .height('100%')
    }
  }
}