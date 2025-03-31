@Entry
@Component
struct App {
  build() {
    List() {
      ForEach(this.orders, (order, i) => {
        OrderItem(order);
      });
    }
    .width('100%')
    .height('100%')
  }
}