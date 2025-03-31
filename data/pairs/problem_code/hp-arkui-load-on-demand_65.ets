@Entry
@Component
struct AlphabetIndexerSample {
  scroller: Scroller = new Scroller()
  private value: string[] = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
  index: number = 0

  build() {
    Row({ direction: FlexDirection.Column }) {
      Row() {
        Navigation()
      }
      Flex({ direction: FlexDirection.Row }) {
        List() {
          ForEach(this.value, (item: string) => {
            ListItem() {
              Text(item)
            }
          }, (item:string)=>item)
        }
        .cachedCount(2)
        .width("100%")
        .height('100%')
      }
    }
  }
}
