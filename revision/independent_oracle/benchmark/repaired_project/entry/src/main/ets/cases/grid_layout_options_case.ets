@Entry
@Component
struct FeaturedGrid {
  private items: number[] = [1, 2, 3, 4];
  private irregularIndexes: number[] = [0, 2];
  private layoutOptions: GridLayoutOptions = {
    regularSize: [1, 1],
    irregularIndexes: this.irregularIndexes
  };

  build() {
    Grid(undefined, this.layoutOptions) {
      LazyForEach(this.items, (item: number) => {
        GridItem() { Text(`${item}`) }
      }, (item: number) => item.toString())
    }
    .cachedCount(2)
    .columnsTemplate('1fr 1fr')
  }
}
