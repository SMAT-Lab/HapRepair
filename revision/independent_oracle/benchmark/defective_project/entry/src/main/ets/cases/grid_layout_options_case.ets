@Entry
@Component
struct FeaturedGrid {
  private items: number[] = [1, 2, 3, 4];

  build() {
    Grid() {
      LazyForEach(this.items, (item: number, index: number) => {
        if (index % 2 === 0) {
          GridItem() { Text(`${item}`) }
            .columnStart(0)
            .columnEnd(1)
        } else {
          GridItem() { Text(`${item}`) }
        }
      }, (item: number) => item.toString())
    }
    .cachedCount(2)
    .columnsTemplate('1fr 1fr')
  }
}
