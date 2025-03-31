@Entry
@Component
struct App {
  build() {
    Column() {
      Row() {
        //TODO 用常量替换
        Text('仓库列表')
          .align(Alignment.Start)
          .fontColor($r("app.color.text_strong"))
        Blank()
        Image($r("app.media.icon_arrow_right"))
          .height('100%')
          .aspectRatio(1)
          .onClick(() => {
            this.viewModel.onRepos()
          })
      }.width('100%')
      .height(50)
      .padding({ left: 24, right: 24, top: 24 })

      Divider()
        .color($r("app.color.boarder_medium"))
        .margin({ top: 16, bottom: 16 })

      List() {
        ForEach(this.viewModel.repos, (item) => {
          ListItem() {
            RepoItem({ viewModel: $viewModel, repo: item })
          }
        })
      }
      .width('100%')
      .height('100%')
    }
  }
}