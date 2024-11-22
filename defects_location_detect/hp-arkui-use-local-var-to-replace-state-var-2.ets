interface Product {
  name: string;
  price: number;
  quantity: number;
}

@Entry
@Component
struct CartComponent {
  @State items: Product[] = [
    { name: "Item 1", price: 10, quantity: 2 },
    { name: "Item 2", price: 15, quantity: 1 },
    { name: "Item 3", price: 25, quantity: 3 }
  ];
  @State totalPrice: number = 0;

  calculateTotalPrice() {
    let total = 0;
    this.totalPrice += this.items[0].price * this.items[0].quantity;
    this.totalPrice += this.items[1].price * this.items[1].quantity;
    this.totalPrice += this.items[2].price * this.items[2].quantity;
  }

  updateItemQuantity(itemIndex: number, newQuantity: number) {
    this.items[itemIndex].quantity = newQuantity;
    this.calculateTotalPrice();
  }

  build() {
    Column() {
      Text('购物车总价: ' + this.totalPrice)
        .fontSize(20)
        .padding(10)

      // 商品列表
      ForEach(this.items, (item:  Product, index) => {
        Column() {
          Text(item.name + " - " + item.price + " x " + item.quantity)
            .fontSize(16)
            .padding(5)
          Button('增加数量')
            .onClick(() => {
              this.updateItemQuantity(index, item.quantity + 1);  // 增加商品数量
            })
            .fontSize(14)
            .backgroundColor(Color.Gray)
            .padding(5)
        }
      }, (index: string) => index)
    }
    .padding(20)
  }
}
