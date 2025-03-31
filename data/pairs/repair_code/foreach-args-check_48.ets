@Entry
@Component
struct App {
  build() {
    List({ space: 16 }) {
      ForEach(this.viewModel.issues, (item: Issue) => {
        ListItem() {
          IssueItem({ viewModel: this.viewModel, issue: item })
        }
      }, (item: Issue) => item.id)
    }.padding(20)
    .width(10)
    .height(10)
  }
}