@Entry
@Component
struct App {
  build() {
    List({ space: 10, initialIndex: 0, scroller: this.scroller }) {
      LazyForEach(this.listValue, (item) => {
        ListItem() {
          Column() {
            Text('     ' + item)
              .height(30)
              .fontSize(16)
              .width('100%')
              .textAlign(TextAlign.Start)
            Column() {
              ForEach(listData, (ele) => {
                if (ele.substr(0, 1) == item.toUpperCase() || ele.substr(0, 1) == item.toLowerCase()) {
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
        .editable(true)
      }, item => item)
    }
    .cachedCount(2)
    .height("100%")
    .width("100%")
  }
}