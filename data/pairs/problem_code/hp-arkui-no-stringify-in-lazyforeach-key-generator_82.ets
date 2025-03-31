@Entry
@Component
struct App {
  build() {
    Row() {
      LazyForEach(this.productList, (product: Product) => {
        ListItem() {
          ProductComponent({ productName: product.name });
        }
      }, (product: Product) => JSON.stringify({ id: product.id, name: product.name }));
    }
  }
}
