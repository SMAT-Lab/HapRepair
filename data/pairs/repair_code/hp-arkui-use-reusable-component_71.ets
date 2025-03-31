
@Reusable
@Component
struct ImageGridItem {
  @State lazyItem: LazyItem<UserFileDataItem> = new LazyItem<UserFileDataItem>(
    new UserFileDataItem("initial_selection", ["arg1", "arg2"], "device123", 0), // 创建 UserFileDataItem 实例
    0
  );
  @State mediaItem: UserFileDataItem | undefined = undefined; // 初始化为 undefined
  @State pageName: string = Constants.PHOTO_TRANSITION_ALBUM; // 直接使用常量初始化
  @State isSelectUpperLimited: boolean = false; // 设置初始值为 false

  constructor(lazyItem: LazyItem<UserFileDataItem>, pageName?: string, isSelectUpperLimited?: boolean) {
    super()
    // 使用 defaults 处理可选参数
    this.lazyItem = lazyItem;
    this.mediaItem = lazyItem.item; // 从 lazyItem 初始化 mediaItem
    // 如果提供了，才使用来自参数的 pageName 和 isSelectUpperLimited
    if (pageName) {
      this.pageName = pageName;
    }
    if (isSelectUpperLimited !== undefined) {
      this.isSelectUpperLimited = isSelectUpperLimited;
    }
  }

  build() {
    if (this.lazyItem && this.mediaItem) {
      ImageGridItemComponent({
        lazyItem: this.lazyItem,
        mediaItem: this.mediaItem,
        pageName: Constants.PHOTO_TRANSITION_ALBUM,
        isSelectUpperLimited: $isSelectUpperLimited
      })
    }
  }
}
@Reusable
@Component
struct ImageGridItemComponent {
  @State lazyItem: LazyItem<UserFileDataItem> = new LazyItem<UserFileDataItem>()
  @State mediaItem: UserFileDataItem | undefined = undefined;
  @State pageName: string = ""; // 直接使用常量初始化
  @State isSelectUpperLimited: boolean = false; // 设置初始值为 false

  build() {
    Grid(this.scroller) {
      LazyForEach(this.groupDataSource, (item: LazyItem<UserFileDataItem>): void => {
        GridItem() {
          ImageGridItem()
        }
        .aspectRatio(1)
        .columnStart(item.get().index % this.gridRowCount)
        .columnEnd(item.get().index % this.gridRowCount)
      }, (item: LazyItem<AlbumDataItem>): string => item.getHashCode())
    }
    .columnsTemplate('1fr '.repeat(this.gridRowCount))
    .columnsGap(Constants.GRID_GUTTER)
    .rowsGap(Constants.GRID_GUTTER)
    .cachedCount(Constants.GRID_CACHE_ROW_COUNT)

    GridScrollBar({ scroller: this.scroller, isHideScrollBar: $isHideScrollBar });
  }
}
