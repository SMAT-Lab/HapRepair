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
      item.getHashCode() : item.item.id)
    }
    .height('100%')
    .width('100%')
    .cachedCount(2)
  }
}