@Entry
@Component
struct App {
  build() {
    List() {
      ForEach(this.orders, (order, i) => {
        OrderItem(order);
      }, (order) => order.orderNumber);
    }
    .width(10)
    .height(10)
  }
}