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
  @Link @Watch('onNotifyChange') profile: ProfileSettings;
  @State localNotificationsEnabled: boolean = true;

  onNotifyChange() {
    this.localNotificationsEnabled = this.profile.notificationsEnabled;
  }

  build() {
    if (this.localNotificationsEnabled) {
      Text('Notifications On')
    } else {
      Text('Notifications Off')
    }
  }
}

@Component
struct AlertComp {
  @Link @Watch('onNotifyChange') profile: ProfileSettings;
  @State localNotificationsEnabled: boolean = true;

  onNotifyChange() {
    this.localNotificationsEnabled = this.profile.notificationsEnabled;
  }

  build() {
    if (this.localNotificationsEnabled) {
      Text('Alerts Enabled')
    } else {
      Text('Alerts Disabled')
    }
  }
}