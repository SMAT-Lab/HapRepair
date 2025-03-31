class Settings {
  theme: string = "dark";
  notificationsEnabled: boolean = true;
}

@Component
struct UserProfile {
  @State userSettings: Settings = new Settings();
  @State userName: string = "John Doe";

  build() {
    Column() {
      Text("User: " + this.userName)
        .fontSize(18)
      Toggle("Enable Notifications", this.userSettings.notificationsEnabled)
        .onChange((value) => {
          this.userSettings.notificationsEnabled = value;
        })
    }
    .padding(16)
  }
}
