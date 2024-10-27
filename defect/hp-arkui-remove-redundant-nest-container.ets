@Entry
@Component
struct NestedViewComponent {
  @State items: Number[] = Array.from(Array<number>(500), (v, k) => k);  // 500 个数据项

  build() {
    Scroll() {
      Column() {
        ForEach(this.items, (item: Number) => {
          Column() {
            Stack() {
              Column() {
                Stack() {
                  Flex() {
                    Column() {
                      Text(item.toString())
                        .fontSize(14)
                        .backgroundColor(Color.Gray)
                        .padding(10)
                    }
                  }
                }
              }
            }
          }
          .backgroundColor(Color.White)
          .padding(5)
        }, (item: string) => item.toString())
      }
      .width('100%')
      .height('100%')
    }
  }
}