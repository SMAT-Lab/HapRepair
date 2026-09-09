@Entry
@Component
struct FeaturedGrid {
  private items: number[] = [1, 2, 3, 4];
  private layoutOptions: GridLayoutOptions = {
    regularSize: [1, 1],
    irregularIndexes: [0, 2],
    getItemSize: (index: number) => {
      if (index % 2 === 0) {
        return [2, 1];
      }
      return [1, 1];
    }
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
