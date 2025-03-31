@Entry
@Component
struct AlphabetIndexerSample {
  scroller: Scroller = new Scroller()
  private value: string[] = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
  private listValue: string[] = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
  index: number = 0
  selectedColor: string = '#00ffff'
  popupColor: string = '#00ffff'
  selectedBackgroundColor: string = '#ff0000'
  popupBackground: string = '#ff0000'
  usingPopup: boolean = true
  itemSize: number = 13
  location: boolean = true

  build() {
    Row({ direction: FlexDirection.Column }) {
      Row() {
        NavigationBar({ title: 'AlphabetIndexer' })
      }.padding({ left: '3%' })

      Flex({ direction: this.location ? FlexDirection.Row : FlexDirection.RowReverse }) {
        List({ space: 10, initialIndex: 0, scroller: this.scroller }) {
          LazyForEach(this.listValue, (item) => {
            ListItem() {
                Text('     ' + item)
                  .height(30)
                  .fontSize(16)
                  .width('100%')
                  .textAlign(TextAlign.Start)
              }

          })
        }
      }
      .cachedCount(2)
      .width("100%")
      .height('100%')
    }
  }
}