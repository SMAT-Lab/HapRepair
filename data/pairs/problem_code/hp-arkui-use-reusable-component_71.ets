@Entry
@Component
struct App {
  layoutOptions: GridLayoutOptions = {
    regularSize: [1,1]
  }
  build() {
    Grid(this.scroller, this.layoutOptions) {
      LazyForEach(this.groupDataSource, (item: LazyItem<UserFileDataItem>): void => {
        GridItem() {
          ImageGridItemComponent({
            lazyItem: item,
            mediaItem: item.get(),
            pageName: Constants.PHOTO_TRANSITION_ALBUM,
            isSelectUpperLimited: $isSelectUpperLimited
          })
        }
        .aspectRatio(1)
        .columnStart(item.get().index % this.gridRowCount)
        .columnEnd(item.get().index % this.gridRowCount)
      }, (item: LazyItem<AlbumDataItem>): string => item.getHashCode())
    }
    .cachedCount(2)
  }
}