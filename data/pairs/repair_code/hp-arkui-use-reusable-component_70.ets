@Reusable
@Component
struct AlbumGridItemComponent {
  @State item: AlbumDataItem = new AlbumDataItem("1", 0, "", 0, "", 0, 0)
  isBigCard: boolean = false

  build() {
    GridItem() {
      AlbumSelectGridItemNewStyle({
        item: this.item,
        isBigCard: this.isBigCard,
      })
    }
  }
}

@Reusable
@Component
struct AlbumSelectGridItemNewStyle {
  @Prop @Watch("aboutToReuse") item: AlbumDataItem = new AlbumDataItem()
  @Prop @Watch("aboutToReuse") isBigCard: boolean = false

  aboutToReuse(params: any): void {

  }
  build() {
    Grid(this.scroller) {
      LazyForEach(this.albumsDataSource, (item: LazyItem<AlbumDataItem>): void => {
        if (item && item.get()) {

          if (item.index === 0) {
            // 处理第一个专辑项
            AlbumGridItemComponent({
              item: item.get(),
              isBigCard: true,
            }); // 设置网格的起始和结束列
          } else {
            // 处理普通的专辑项
            AlbumGridItemComponent({
              item: item.get(),
              isBigCard: false,
            });
          }
        }
      }, (item: LazyItem<AlbumDataItem>): string => (item != undefined && item != null) &&
        (item.get() != undefined && item.get() != null) ?
      item.getHashCode() : item.id)

    }
    .height('100%')
    .width('100%')
    .cachedCount(2)
  }
}