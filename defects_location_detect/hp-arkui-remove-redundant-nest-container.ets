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
              Stack() {
                Stack() {
                  Flex() {
                    Column() {
                      Text(item.toString())
                        .fontSize(14)
                        .backgroundColor(Color.Gray)
                        .padding(10)
                    }
                    .justifyContent(1)
                  }
                  .onClick(()=>{
                    console.log("123")
                  })
                  .width("100%")
                }
                .height("100%")
              }
              .margin(1)
            }
            .padding(1)
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