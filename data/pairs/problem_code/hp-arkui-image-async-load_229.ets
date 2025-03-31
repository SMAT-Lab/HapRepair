@Entry
@Component
struct LargeImageComponent {
  build() {
    Column() {
      // 本地高清图片 hd_image.png
      Image($r('app.media.hd_image'))
        .width('100%')
        .height(200)
        .syncLoad(true) // 同步加载大图片
    }
  }
}
