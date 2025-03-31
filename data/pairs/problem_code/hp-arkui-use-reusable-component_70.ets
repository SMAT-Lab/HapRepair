@Entry
@Component
struct App {
  layoutOption: GridLayoutOptions =  {
    regularSize: [1, 1]
  }
  build() {
    Grid(this.scroller, this.layoutOption) {
      LazyForEach(this.albumsDataSource, (item: LazyItem<AlbumDataItem>): void => {
        if ((item != undefined && item != null) && (item.get() != undefined && item.get() != null)) {
          if (item.index === 0) {
            GridItem() {
              AlbumSelectGridItemNewStyle({
                item: item.get(),
                isBigCard: true,
              })
            }.columnStart(0).columnEnd(1)
          } else if (item != null) {
            GridItem() {
              AlbumSelectGridItemNewStyle({
                item: item.get(),
                isBigCard: false,
              })
            }
          }
        }
      }, (item: LazyItem<AlbumDataItem>): string => (item != undefined && item != null) &&
        (item.get() != undefined && item.get() != null) ?
      item.getHashCode() : item.id)
    }
    .cachedCount(2)
  }
}