import { MyDataSource } from './MyDataSource';
@Entry
@Component
struct MyComponent {
  private arr: MyDataSource = new MyDataSource()

  build() {
    List() {
      // List中建议使用LazyForEach
      LazyForEach(this.arr, (item: string) => {
        ListItem() {
          Text(`item value: ${item}`)
        }
      }, (item: number) => item.toString())
    }
    .width('100%')
    .height('100%')
    .cachedCount(2)
  }
}