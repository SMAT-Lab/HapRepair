@Entry
@Component
struct PhotoGallery {
  private photos: string[] = [];
  private irregularData: number[] = [];
  private layoutOptions: GridLayoutOptions = {
    regularSize: [1, 1],
    irregularIndexes: this.irregularData,
    getItemSize: (index: number) => {
      if (this.irregularData.includes(index)) {
        return [3, 1];
      }
      return [1, 1];
    }
  };

  aboutToAppear() {
    for (let i = 1; i <= 500; i++) {
      this.photos.push('Photo ' + i);
      if ((i) % 7 === 0) {
        this.irregularData.push(i - 1);
      }
    }
  }

  build() {
    Column() {
      Grid(undefined, this.layoutOptions) {
        LazyForEach(this.photos, (photo: string) => {
          GridItem() {
            Image($r('app.media.' + photo))
              .width('100%')
              .height(100)
          }
        }, (photo: string) => photo)
      }
      .columnsTemplate('1fr 1fr 1fr')
      .columnsGap(5)
      .rowsGap(5)
      .height('100%')
      .width('100%')
      .cachedCount(2)
    }
  }
}
