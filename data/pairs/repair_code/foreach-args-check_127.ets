@Entry
@Component
struct App {
  build() {
    Grid() {
      ForEach(this.comments, (comment, idx) => {
        CommentView(comment);
      }, (comment) => comment.commentID);
    }
  }
}