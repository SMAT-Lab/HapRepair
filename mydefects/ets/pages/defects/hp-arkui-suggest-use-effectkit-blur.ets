@Entry
@Component
struct BlurEffectComponent {
  build() {
    Column({ space: 10 }) {
      Text('Text with blur effect').fontSize(18).fontColor(0xAAAAAA).width('90%')
      Text('This is blurred text')
        .fontSize(20)
        .width('90%')
        .height(50)
        .backdropBlur(3)
        .backgroundImage('/pages/attrs/image/image.jpg')
        .backgroundImageSize({ width: 1200, height: 160 })

      Text('Background blur effect').fontSize(18).fontColor(0xAAAAAA).width('90%')
      Image('resources/base/media/background.jpg')
        .blur(1)
        .width('90%')
        .height(200)
    }
    .width('100%')
    .margin({ top: 10 })
  }
}