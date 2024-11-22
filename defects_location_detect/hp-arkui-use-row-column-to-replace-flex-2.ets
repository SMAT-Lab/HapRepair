@Entry
@Component
struct NewsListComponent {
  build() {
    Flex({ direction: FlexDirection.Column }) {
      Text('Top News')
        .fontSize(24)
        .fontWeight('bold')
        .padding(10)

      Flex({ direction: FlexDirection.Column }) {
        // 新闻项
        Text('Breaking News: Major event in the city')
          .fontSize(16)
          .padding(10)
        Text('Politics: Latest updates on the election')
          .fontSize(16)
          .padding(10)
        Text('Sports: A new record set by a local athlete')
          .fontSize(16)
          .padding(10)
      }
      .width('100%')
      .padding(10)

      Flex({ direction: FlexDirection.Row }) {
        Text('Share')
          .fontSize(14)
          .padding(10)
        Button('Read More')
          .padding(10)
      }
      .width('100%')
      .padding(10)
    }
    .padding(20)
  }
}
