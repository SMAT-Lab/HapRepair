@Reusable
@Component
struct ListItemComponent {
  item: string = ''
  @State listData: string[] = []

  build() {
    Column() {
      Text('     ' + this.item)
        .height(30)
        .fontSize(16)
        .width('100%')
        .textAlign(TextAlign.Start)

      // 这里使用一个内部 ForEach 遍历 listData
      ForEach(this.listData, (ele: string) => {
        if (ele.substr(0, 1) === this.item.toUpperCase() || ele.substr(0, 1) === this.item.toLowerCase()) {
          Text('' + ele)
            .fontSize(16)
            .width('100%')
            .textAlign(TextAlign.Center)
            .margin({ top: 5, bottom: 5 })
            .height(72)
            .borderRadius(24)
            .backgroundColor('#ffffff')
        }
      }, (item: string)=>item)
    }
  }
}

class FruitValue implements IDataSource {
  public fruit: string[] = ["A", "B", "C", "D"]
  totalCount(): number {
    return this.fruit.length
  }

  getData(index: number): string {
    return this.fruit[index]
  }

  registerDataChangeListener(listener: DataChangeListener): void {
    throw new Error('Method not implemented.')
  }

  unregisterDataChangeListener(listener: DataChangeListener): void {
    throw new Error('Method not implemented.')
  }

}

@Entry
@Component
struct MyComponent {
  private listValue: FruitValue = new FruitValue()
  private listData: string[] = ["Apple", "Avocado", "Banana", "Blueberry", "Cherry", "Date"]

  build() {
    List() {
      LazyForEach(this.listValue, (item: string) => {
        ListItem() {
          // 使用 reusable component 传递 item 和 listData
          ListItemComponent({
            item: item,
            listData: this.listData
          })
        }
      }, (item: string) => item)
    }
    .height('100%')
    .width('100%')
    .cachedCount(2)
  }
}
