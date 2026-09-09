@Entry
@Component
struct PosterPreview {
  build() {
    Image($r('app.media.large_poster'))
      .width(180)
      .height(240)
  }
}
