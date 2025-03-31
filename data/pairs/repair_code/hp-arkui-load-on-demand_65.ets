export class DataSource implements IDataSource {
  private dataArray: string[] = []

  constructor() {
    for(let i = 0; i < 26; i++) {
      this.dataArray.push(String.fromCharCode(65 + i)) // A - Z
    }
  }

  registerDataChangeListener(listener: DataChangeListener): void {
    throw new Error("Method not implemented.")
  }

  unregisterDataChangeListener(listener: DataChangeListener): void {
    throw new Error("Method not implemented.")
  }

  public getData(index: number): string {
    return this.dataArray[index]
  }

  public totalCount(): number {
    return this.dataArray.length
  }
}

@Entry
@Component
struct AlphabetIndexerSample {
  private value: DataSource = new DataSource()

  build() {
    Column({ direction: FlexDirection.Column }) {
      Row() {
        Navigation() {
          
        }
      }
      Flex({ direction: FlexDirection.Row }) {
        List() {
          LazyForEach(this.value, (item: string) => {
            ListItem() {
              Text(item)
            }
          })
        }
        .cachedCount(2)
        .width("100%")
        .height("100%")
      }
    }
  }
}
