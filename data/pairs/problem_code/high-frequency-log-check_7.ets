@Entry
@Component
struct App {
  build() {
    Stack() {
      Column() {
        LoadingPanel()
      }
      .width('100%')
      .height('100%')

      Row() {
        if (this.thumbnail !== null && this.thumbnail !== undefined) {
          Image(this.thumbnail)
            .rotate({
              x: 0,
              y: 0,
              z: 1,
              angle: 0
            })
            .onComplete((): void => {
              Log.info(TAG,
                'onComplete finish, index: ' + this.item.index + ', item: ' + JSON.stringify(this.item) + ', uri: ' +
                this.thumbnail + '.');
            })
            .onError((): void => {
              Log.error(TAG, 'image show error ' + this.thumbnail + ' ' + this.item.width + ' ' + this.item.height);
            })
        }

      }
      .width('100%')
      .height('100%')
    }
  }
}