
@Entry
@Component
struct GridTest {
  private items: number[] = [10, 20, 30, 40, 50];

  build() {
    RelativeContainer() {
      List() {
        ForEach(this.items, (item: number, index: number) => {
          ListItem() {
            Text(item.toString());
          }
        })
      }
      .width('100%')
      .height('100%')
    }
    .height('100%')
    .width('100%')
  }
}
