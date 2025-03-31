@Entry
@Component
struct App {
  build() {
    Row() {
      ForEach(this.notifications, (note) => {
        NotificationItem(note);
      });
    }
  }
}