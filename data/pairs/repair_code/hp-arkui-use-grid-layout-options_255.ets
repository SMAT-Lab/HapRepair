@Entry
@Component
struct ProductGrid {
  private products: string[] = [];
  private irregularData: number[] = [];
  private layoutOptions: GridLayoutOptions = {
    regularSize: [1, 1],
    irregularIndexes: this.irregularData,
    getItemSize: (index: number) => {
      if (this.irregularData.includes(index)) {
        return [2, 1];
      }
      return [1, 1];
    }
  };

  aboutToAppear() {
    for (let i = 1; i <= 1000; i++) {
      this.products.push('Product ' + i);
      if ((i - 1) % 5 === 0) {
        this.irregularData.push(i - 1);
      }
    }
  }

  build() {
    Column() {
      Grid(undefined, this.layoutOptions) {
        LazyForEach(this.products, (item: string) => {
          GridItem() {
            Text(item)
              .fontSize(16)
              .width('100%')
              .height(80)
              .backgroundColor(Color.White)
          }
        }, (item: string) => item)
      }
      .columnsTemplate('1fr 1fr 1fr')
      .columnsGap(10)
      .rowsGap(10)
      .height('100%')
      .width('100%')
      .cachedCount(2)
    }
  }
}
