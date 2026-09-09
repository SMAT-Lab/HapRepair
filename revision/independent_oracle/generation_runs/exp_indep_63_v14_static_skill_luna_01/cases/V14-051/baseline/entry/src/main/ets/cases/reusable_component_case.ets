@Component
struct ProductTile {
  @State label: string = '';
  aboutToReuse(params: Record<string, Object>): void {
    this.label = params.label as string;
  }
  build() { Text(this.label) }
}

@Entry
@Component
struct ProductGrid {
  private products: string[] = [];
  build() {
    Grid() {
      LazyForEach(this.products, (product: string) => {
        GridItem() { ProductTile({ label: product }) }
      }, (product: string) => product)
    }
  }
}
