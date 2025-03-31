@Entry
@Component
struct PhotoGallery {
  private photos: string[] = [];

  aboutToAppear() {
    for (let i = 1; i <= 500; i++) {
      this.photos.push('Photo ' + i);
    }
  }

  build() {
    Column() {
      Grid() {
        LazyForEach(this.photos, (photo: string, index: number) => {
          if ((index + 1) % 7 === 0) {
            GridItem() {
              Image($r('app.media.' + photo))
                .width('100%')
                .height(200)
            }
            .columnStart(0)
            .columnEnd(4)
          } else {
            GridItem() {
              Image($r('app.media.' + photo))
                .width('100%')
                .height(100)
            }
          }
        }, (photo: string) => photo)
      }
      .columnsTemplate('1fr 1fr 1fr')
      .columnsGap(5)
      .rowsGap(5)
      .cachedCount(2)
      .width("100%")
      .height('100%')
    }
  }
}
