@Entry
@Component
struct App {
  build() {
    Row() {
      LazyForEach(this.notificationList, (notification: Notification) => {
        ListItem() {
          NotificationComponent({ message: notification.message });
        }
      }, (notification: Notification) => JSON.stringify(notification.details));
    }
  }
}
