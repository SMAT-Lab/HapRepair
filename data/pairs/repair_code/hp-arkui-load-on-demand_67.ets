export class MyStringDataSource implements IDataSource {
  registerDataChangeListener(listener: DataChangeListener): void {
    throw new Error("Method not implemented.")
  }

  unregisterDataChangeListener(listener: DataChangeListener): void {
    throw new Error("Method not implemented.")
  }
  private items: string[] = ["Apple", "Banana", "Cherry", "Date"]

  public getData(index: number): string {
    return this.items[index]
  }

  public totalCount(): number {
    return this.items.length
  }
}

@Entry
@Component
struct MyComponent {
  private dataSource: MyStringDataSource = new MyStringDataSource()

  build() {
    List() {
      LazyForEach(this.dataSource, (item: string) => {
        ListItem() {
          Text(item)
        }
      })
    }
    .width("100%")
    .height("100%")
    .cachedCount(2)
  }
}
