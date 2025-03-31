@Entry
@Component
struct ShoppingCart {
  @State totalPrice: number = 0;

  addItem(price: number) {
    let total = this.totalPrice;
    total += price;
    total += this.calculateTax(price);
    total += this.calculateShipping(price);
    this.totalPrice = total;
  }

  calculateTax(price: number): number {
    return price * 0.1;
  }

  calculateShipping(price: number): number {
    return 5;
  }

  build() {
    Column() {
      Text('总价：' + this.totalPrice)
        .fontSize(18)
      Button('添加商品')
        .onClick(() => this.addItem(100))
    }
  }
}
