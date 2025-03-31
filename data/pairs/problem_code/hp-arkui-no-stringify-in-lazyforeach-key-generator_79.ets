@Entry
@Component
struct App {
  build() {
    Grid(this.scroller) {
      LazyForEach(this.albumsDataSource, (item: LazyItem<AlbumDataItem>): void => {
        if (item && item.get()) {
          Text()
        }
      }, (item: LazyItem<AlbumDataItem>): string => (item != undefined && item != null) &&
        (item.get() != undefined && item.get() != null) ?
      item.getHashCode() : JSON.stringify(item))
    }
    .cachedCount(2)
    .width("100%")
    .height('100%')
  }
}