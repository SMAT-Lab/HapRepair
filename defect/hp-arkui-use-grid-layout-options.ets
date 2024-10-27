@Reusable
@Component
struct ImageItem {

  build() {
    // 显示图片的组件
    Image("image_clicked.png")
      .width('100%')
      .height(100)
      .border({ width: 2, color: 0xF9CF93 })
  }
}

@Entry
@Component
struct PhotoGalleryComponent {
  private imageDataSource: Array<string> = [];

  // 初始化数据源，包含2000个图片URL
  aboutToAppear() {
    for (let i = 1; i <= 2000; i++) {
      this.imageDataSource.push(`image_${i}.png`);
    }
  }

  build() {
    Column() {
      Text('Image Gallery').fontSize(20).padding(10)

      // 使用 Grid 来显示图片列表
      Grid() {
        LazyForEach(this.imageDataSource, (imageUrl: string, index: number) => {
          GridItem() {
            ImageItem()
          }
          .columnStart(0)
          .columnEnd(2)
        }, (imageUrl: string) => imageUrl)
      }
      .cachedCount(10)
      .columnsTemplate('1fr 1fr 1fr 1fr 1fr')
      .columnsGap(10)
      .rowsGap(10)
      .width('100%')
      .height('80%')
    }
    .width('100%')
    .padding(10)
  }
}
