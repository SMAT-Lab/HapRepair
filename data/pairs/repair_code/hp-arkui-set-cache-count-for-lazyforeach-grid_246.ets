@Entry
@Component
struct ProductGrid {
  private products: number[] = [];

  aboutToAppear() {
    for (let i = 1; i <= 1000; i++) {
      this.products.push(i);
    }
  }

  build() {
    Column() {
      Grid() {
        LazyForEach(this.products, (product: number) => {
          GridItem() {
            // 自定义组件，显示产品信息
            Text('产品编号：' + product)
              .fontSize(16)
              .width('100%')
              .height(80)
              .textAlign(TextAlign.Center)
          }
        }, (product: number) => product.toString())
      }
      // 设置 GridItem 的缓存数量
      .cachedCount(3)
      .columnsTemplate('1fr 1fr 1fr')
      .columnsGap(5)
      .rowsGap(5)
      .height(600)
    }
  }
}