@Entry
@Component
struct App {
  build() {
    Stack() {
      ForEach(this.books, (book) => {
        BookItem(book);
      });
    }
  }
}