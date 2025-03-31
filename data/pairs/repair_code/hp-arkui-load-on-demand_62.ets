export class DataSource implements IDataSource{
  private dataArray: string[] = []
  private listeners: DataChangeListener[] = []

  constructor() {
    for(let i=0; i< 26; i++) {
      this.dataArray.push(i + 'a')
    }
  }

  public getData(index: number): string {
    return this.dataArray[index]
  }

  public totalCount(): number {
    return this.dataArray.length
  }

  registerDataChangeListener(listener: DataChangeListener): void {
    if (this.listeners.indexOf(listener) < 0) {
      this.listeners.push(listener)
    }
  }

  unregisterDataChangeListener(listener: DataChangeListener): void {
    const pos = this.listeners.indexOf(listener)
    if (pos >= 0) {
      this.listeners.splice(pos, 1)
    }
  }
}
@Entry
@Component
struct AlphabetIndexerSample {
  scroller: Scroller = new Scroller()
  private value: string[] = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
  // private listValue: string[] = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
  private listValue: DataSource = new DataSource()
  index: number = 0
  selectedColor: string = '#00ffff'
  popupColor: string = '#00ffff'
  selectedBackgroundColor: string = '#ff0000'
  popupBackground: string = '#ff0000'
  usingPopup: boolean = true
  itemSize: number = 13
  location: boolean = true

  build() {
    Column({ direction: FlexDirection.Column }) {
      Row() {
        NavigationBar({ title: 'AlphabetIndexer' })
      }.padding({ left: '3%' })

      Flex({ direction: this.location ? FlexDirection.Row : FlexDirection.RowReverse }) {
        List({ space: 10, initialIndex: 0, scroller: this.scroller }) {
          LazyForEach(this.listValue, (item: string) => {
            ListItem() {
                Text('     ' + item)
                  .height(30)
                  .fontSize(16)
                  .width('100%')
                  .textAlign(TextAlign.Start)

            }
          })
        }
        .width("100%")
        .height("100%")
        .cachedCount(2)
      }
    }
  }
}