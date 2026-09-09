@Entry
@Component
struct ThumbnailGrid {
  private thumbnails: number[] = [1, 2, 3, 4, 5, 6];

  build() {
    Grid() {
      LazyForEach(this.thumbnails, (thumbnail: number) => {
        GridItem() {
          Text(`${thumbnail}`)
        }
      }, (thumbnail: number) => thumbnail.toString())
    }
    .cachedCount(1)
    .columnsTemplate('1fr 1fr')
  }
}
