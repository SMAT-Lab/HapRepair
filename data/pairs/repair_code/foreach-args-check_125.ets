@Entry
@Component
struct App {
  build() {
    Column() {
      ForEach(this.products, (product: Product) => {
      }, (product: Product) => product.productCode)
    }
  }
}