@Entry
@Component
struct App {
  build() {
    Row() {
      LazyForEach(this.productList, (product: Product) => {
        ListItem() {
          ProductComponent({ productName: product.name });
        }
      }, (product: Product) => product.id.toString());
    }
  }
}