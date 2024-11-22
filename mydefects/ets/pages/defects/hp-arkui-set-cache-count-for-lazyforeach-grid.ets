@Entry
@Component
struct ProductListComponent {
  @State products: number[] = Array.from<number, number>(Array(1000), (_, i) => i + 1);

  build() {
    Column() {
      Text('产品列表')
        .fontSize(24)
        .padding(10)
        .textAlign(TextAlign.Center)

      // 产品列表
      List() {
        LazyForEach(this.products, (item: number) => {
          ListItem() {
            // 使用 ProductItem 自定义组件，嵌套显示产品信息
            ProductItem({ productId: item })
          }
        }, (item: number) => item.toString())
      }
      .height(500)
      .width('100%')
      .backgroundColor(Color.LightGray)
    }
  }
}

// 定义的产品组件，用于显示产品的基本信息
@Reusable
@Component
struct ProductItem {
  @State productId: number = 0;

  aboutToReuse(params: ESObject): void {
    this.productId = params.productId as number;
  }

  build() {
    Row() {
      Text('产品ID: ' + this.productId)
        .fontSize(18)
        .padding(10)
    }
    .backgroundColor(0xE5E5E5)
    .padding(5)
  }
}