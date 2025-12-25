@Entry
@Component
struct LazyGridCachedCountBad {
  private items: number[] = [1, 2, 3, 4, 5];

  build() {
    Column() {
      // Grid + LazyForEach without cachedCount: should trigger
      // @performance/hp-arkui-set-cache-count-for-lazyforeach-grid.
      Grid().cachedCount(4) {
        LazyForEach(this.items, (item: number, index?: number) => {
          GridItem() {
            Text(`${item}`)
          }
        }, (item) => item)
      }
    }
  }
}
