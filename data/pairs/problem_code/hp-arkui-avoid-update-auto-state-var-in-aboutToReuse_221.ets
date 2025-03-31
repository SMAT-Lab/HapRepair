@Reusable
@Component
struct ProductItem {
  @State name: string = '';
  @Link price: number;

  aboutToReuse(params: Record<string, Object>): void {
    this.name = params.name as string;
    this.price = params.price as number;
  }

  build() {
    Column() {
      Text('产品名称：' + this.name)
      Text('价格：' + this.price)
    }
  }
}

interface GeneratedTypeLiteralInterface_1 {
  name: string;
  price: number;
}

@Entry
@Component
struct ProductList {
  private products: Array<GeneratedTypeLiteralInterface_1> = [];

  aboutToAppear(): void {
    // 初始化产品列表
    this.products = [
      { name: '商品A', price: 100 },
      { name: '商品B', price: 200 },
    ];
  }

  build() {
    Column() {
      List() {
        ForEach(this.products, (item: GeneratedTypeLiteralInterface_1) => {
          ListItem() {
            ProductItem({ name: item.name, price: item.price })
          }
        }, (item:GeneratedTypeLiteralInterface_1)=>item.name)
      }
      .cachedCount(2)
      .width("100%")
      .height('100%')
    }
  }
}
