export class MyDataSource implements IDataSource {
  registerDataChangeListener(listener: DataChangeListener): void {
    throw new Error("Method not implemented.")
  }

  unregisterDataChangeListener(listener: DataChangeListener): void {
    throw new Error("Method not implemented.")
  }

  private items: number[] = [] //  = Array.from({ length: 1000 }, (_, index) => index)

  constructor() {
    for(let i=0; i< 1000; i++) {
      this.items.push(i)
    }
  }
  public getData(index: number): number {
    return this.items[index]
  }

  public totalCount(): number {
    return this.items.length
  }
}

@Entry
@Component
struct MyComponent {
  private dataSource: MyDataSource = new MyDataSource()

  build() {
    List() {
      LazyForEach(this.dataSource, (item: number) => {
        ListItem() {
          Text(`Item: ${item}`)
        }
      })
    }
    .width("100%")
    .height("100%")
    .cachedCount(2)
  }
}
