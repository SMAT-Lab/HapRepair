
// ... (no changes in this part)

  build() {
    Column() {
      ActionBar({
        actionBarProp: $actionBarProp,
        onMenuClicked: (action: Action): void => this.onMenuClicked(action)
      })

      if (this.isEmpty) {
        NoPhotoComponent({ title: $r('app.string.no_distributed_photo_head_title_album') })
      } else {
        Stack() {
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
            }, (item: LazyItem<UserFileDataItem>): string => item.getHashCode())
          }
          .columnsTemplate('1fr '.repeat(this.gridRowCount))
          .columnsGap(Constants.GRID_GUTTER)
          .rowsGap(Constants.GRID_GUTTER)
          .cachedCount(2)

          if (this.scroller !== null && this.scroller !== undefined) {
            GridScrollBar({ scroller: this.scroller, isHideScrollBar: $isHideScrollBar });
          }
        }.layoutWeight(1)
        .padding({
          bottom: ((this.isSelectedMode || (this.albumInfo.id === MediaConstants.ALBUM_ID_RECYCLE)) && !this.isHorizontal) ? Constants.ActionBarHeight : 0
        })
      }

      CustomDialogView();
      if (this.isSelectedMode || (this.albumInfo.id === MediaConstants.ALBUM_ID_RECYCLE) && !this.isEmpty) {
        ToolBar({
          toolMenuList: $toolBarMenuList,
          onMenuClicked: (action: Action): void => this.onMenuClicked(action)
        })
      }
    }
    .backgroundColor($r('app.color.default_background_color'))
    .padding({
      top: px2vp(this.leftBlank[1]),
      bottom: px2vp(this.leftBlank[3])
    })
  }

  private layoutOptions: GridLayoutOptions = {
    regularSize: [1, 1]
  };
  pageTransition(): void {
    PageTransitionEnter({ type: RouteType.None, duration: 1 })
      .opacity(0)
    PageTransitionExit({ type: RouteType.None, duration: 1 })
      .opacity(0)
  }
}