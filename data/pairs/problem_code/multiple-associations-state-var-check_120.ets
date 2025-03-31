@Observed
class ProfileSettings {
  notificationsEnabled: boolean = true;
}

@Entry
@Component
struct DemoF {
  @State profile: ProfileSettings = new ProfileSettings();

  build() {
    Row() {
      NotificationComp({ profile: this.profile })
      AlertComp({ profile: this.profile })
      Text('Toggle Notifications')
        .onClick(() => {
          this.profile.notificationsEnabled = !this.profile.notificationsEnabled;
        })
    }
  }
}

@Component
struct NotificationComp {
  @Link profile: ProfileSettings;

  build() {
    if (this.profile.notificationsEnabled) {
      Text('Notifications On')
    } else {
      Text('Notifications Off')
    }
  }
}

@Component
struct AlertComp {
  @Link profile: ProfileSettings;

  build() {
    if (this.profile.notificationsEnabled) {
      Text('Alerts Enabled')
    } else {
      Text('Alerts Disabled')
    }
  }
}