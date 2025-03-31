@Entry
@Component
struct ProductGrid {
  private products: string[] = [];

  aboutToAppear() {
    for (let i = 1; i <= 1000; i++) {
      this.products.push('Product ' + i);
    }
  }

  build() {
    Column() {
      Grid() {
        LazyForEach(this.products, (item: string, index: number) => {
          if (index % 5 === 0) {
            GridItem() {
              Text(item)
                .fontSize(16)
                .width('100%')
                .height(120)
                .backgroundColor(Color.LightGray)
            }
            .columnStart(0)
            .columnEnd(3)
          } else {
            GridItem() {
              Text(item)
                .fontSize(16)
                .width('100%')
                .height(80)
                .backgroundColor(Color.White)
            }
          }
        }, (item: string) => item)
      }
      .columnsTemplate('1fr 1fr 1fr')
      .columnsGap(10)
      .rowsGap(10)
      .cachedCount(2)
      .width("100%")
      .height('100%')
    }
  }
}
