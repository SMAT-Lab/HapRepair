// 文件：ImageGallery.ets
@Entry
@Component
struct ImageGallery {
  private imageList: string[] = [];

  aboutToAppear() {
    for (let i = 1; i <= 500; i++) {
      this.imageList.push('image_' + i + '.png');
    }
  }

  build() {
    Column() {
      Grid() {
        LazyForEach(this.imageList, (image: string) => {
          GridItem() {
            Image($r('app.media.' + image))
              .width(100)
              .height(100)
          }
        }, (image: string) => image)
      }
      // 未设置缓存数量
      .columnsTemplate('1fr 1fr 1fr 1fr')
      .columnsGap(5)
      .rowsGap(5)
      .height('100%')
    }
  }
}
