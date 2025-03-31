@Entry
@Component
struct App {
  build() {
    List({ space: 16 }) {
      ForEach(this.viewModel.issues, (item) => {
        ListItem() {
          IssueItem({ viewModel: this.viewModel, issue: item })
        }
      })
    }.padding(20)
    .width('100%')
    .height('100%')
  }
}