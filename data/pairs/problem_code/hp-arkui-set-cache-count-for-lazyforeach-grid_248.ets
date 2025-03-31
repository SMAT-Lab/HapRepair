// 文件：ArticleGrid.ets
@Entry
@Component
struct ArticleGrid {
  private articles: { id: number; title: string }[] = [];

  aboutToAppear() {
    for (let i = 1; i <= 800; i++) {
      this.articles.push({ id: i, title: '文章标题 ' + i });
    }
  }

  build() {
    Column() {
      Grid() {
        LazyForEach(this.articles, (article) => {
          GridItem() {
            Text(article.title)
              .fontSize(14)
              .width('100%')
              .height(60)
          }
        }, (article) => article.id.toString())
      }
      // 未设置缓存数量
      .columnsTemplate('1fr 1fr')
      .columnsGap(10)
      .rowsGap(10)
      .height(500)
    }
  }
}
