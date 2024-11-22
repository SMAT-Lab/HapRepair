@Entry
@Component
struct LargeListComponent {
  @State items: string[] = Array.from<number, string>(
    { length: 100 },
    (_, i) => `Item ${i + 1}`
  );

  build() {
    List() {
      ForEach(this.items, (item: string) => {
        ListItem() {
          Text(item)
        }
      }, (item: string) => item)
    }
    .width('100%')
    .height('100%')
  }
}
