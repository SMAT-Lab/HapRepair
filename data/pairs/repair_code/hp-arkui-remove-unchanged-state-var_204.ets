class User {
  id: number = 1;
  email: string = "user@example.com";
}

@Component
struct Dashboard {
  @State currentUser: User = new User();
  // welcomeMessage 未发生变化，改为普通变量
  welcomeMessage: string = "Welcome to the Dashboard";

  build() {
    Column() {
      Text(this.welcomeMessage)
        .fontSize(22)
      // 显示用户信息
      UserInfo(this.currentUser)
    }
    .padding(10)
  }
}
