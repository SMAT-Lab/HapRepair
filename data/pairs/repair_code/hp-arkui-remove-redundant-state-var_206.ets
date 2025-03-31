@Entry
@Component
struct NotificationComponent {
  // notificationCount 未在 UI 中使用，移除状态变量
  notificationCount: number = 0;

  incrementNotifications() {
    this.notificationCount += 1;
    // 逻辑处理，例如发送通知
  }

  doSomething() {
    // 执行其他逻辑
  }

  build() {
    Column() {
      Button("Send Notification")
        .onClick(() => this.doSomething())
      // notificationCount 未在 UI 中使用
    }
    .padding(10)
  }
}
