@Entry
@Component
struct ImageGallery {
  build() {
    Scroll() {
      Flex() {
        Image($r('app.media.image1'))
        Image($r('app.media.image2'))
      }
    }
  }
}
