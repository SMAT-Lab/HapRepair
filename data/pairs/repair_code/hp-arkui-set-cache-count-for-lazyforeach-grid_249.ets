// 文件：VideoGrid.ets
@Entry
@Component
struct VideoGrid {
  private videos: string[] = [];

  aboutToAppear() {
    for (let i = 1; i <= 600; i++) {
      this.videos.push('video_' + i);
    }
  }

  build() {
    Column() {
      Grid() {
        LazyForEach(this.videos, (video: string) => {
          GridItem() {
            // 显示视频缩略图
            Image($r('app.media.' + video + '_thumb'))
              .width(150)
              .height(100)
          }
        }, (video: string) => video)
      }
      // 设置缓存数量
      .cachedCount(3)
      .columnsTemplate('1fr 1fr 1fr')
      .columnsGap(8)
      .rowsGap(8)
      .height(600)
    }
  }
}
