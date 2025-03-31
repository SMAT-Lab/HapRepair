@Entry
@Component
struct ListManager {
  @State items: string[] = [];

  addItems(newItems: string[]) {
    this.items.push(newItems[0]);
    this.items.push(newItems[1]);
    this.items.push(newItems[2]);
  }

  build() {
    Column() {
      ForEach(this.items, (item) => {
        Text(item)
      })
      Button('添加项目')
        .onClick(() => this.addItems(['Item A', 'Item B', 'Item C']))
    }
  }
}
