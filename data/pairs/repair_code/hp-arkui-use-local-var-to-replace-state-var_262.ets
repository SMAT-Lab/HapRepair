@Entry
@Component
struct ListManager {
  @State items: string[] = [];

  addItems(newItems: string[]) {
    let tempItems = [...this.items];
    tempItems.push(newItems[0]);
    tempItems.push(newItems[1]);
    tempItems.push(newItems[2]);
    this.items = tempItems;
  }

  build() {
    Column() {
      ForEach(this.items, (item) => {
        Text(item)
      }, (item:string)=>item)
      Button('添加项目')
        .onClick(() => this.addItems(['Item A', 'Item B', 'Item C']))
    }
  }
}
