@Reusable
@Component
struct ProductItem {
  @State name: string = '';
  @State price: number = 0;
  prices: Map<string, number> = new Map<string, number>()

  aboutToReuse(params: Record<string, Object>): void {
    this.name = params.name as string;
    const prices = params.prices as Map<string, number>
    this.price = prices.get(this.name) || 0
  }

  build() {
    Column() {
      Text('产品名称：' + this.name)
      Text('价格：' + this.price)
    }
  }
}

@Entry
@Component
struct ProductList {
  private products: string[] = ['商品A', '商品B', '商品C'];
  private prices: Map<string, number> = new Map();

  // 模拟一个耗时的价格计算函数
  calculatePrice(productName: string): number {
    let price = 0;
    for (let i = 0; i < 100000; i++) {
      price += i;
    }
    return price;
  }

  aboutToAppear() {
    this.products.forEach((product) => {
      const price = this.calculatePrice(product);
      this.prices.set(product, price);
    });
  }

  build() {
    Column() {
      ForEach(this.products, (product: string) => {
        ProductItem({ name: product, prices: this.prices})
      }, (product: string) => product)
    }
  }
}
