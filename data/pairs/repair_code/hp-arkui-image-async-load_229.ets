@Entry
@Component
struct LargeImageComponent {
  build() {
    Column() {
      // 本地高清图片 hd_image.png
      Image($r('app.media.hd_image'))
        .width('100%')
        .height(200)
        // 移除 syncLoad，使用异步加载
    }
  }
}
