@Entry
@Component
struct ExampleImageComponent {
  build() {
    Column() {
      Image($r('app.media.large_image'))
        .border({ width: 2 })
        .borderStyle(BorderStyle.Solid)
        .height(200)
        .width(200)
        .syncLoad(true)
    }
  }
}