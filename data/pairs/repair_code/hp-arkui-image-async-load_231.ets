@Entry
@Component
struct BackgroundImageComponent {
  build() {
    Stack() {
      // 网络大图片
      Image('https://example.com/large-background.jpg')
        .width('100%')
        .height('100%')
        // 移除 syncLoad，使用异步加载
      Text('欢迎使用我们的应用')
        .fontSize(24)
        .fontColor(Color.White)
    }
  }
}
