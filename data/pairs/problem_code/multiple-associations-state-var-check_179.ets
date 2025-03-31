@Observed
class NotificationSettings {
  enableSound: boolean = true;
}

@Entry
@Component
struct DemoE {
  @State settings: NotificationSettings = new NotificationSettings();

  build() {
    Column() {
      SoundControl({ settings: this.settings })
      VibrationControl({ settings: this.settings })
      Text('Toggle Sound')
        .onClick(() => {
          this.settings.enableSound = !this.settings.enableSound;
        })
    }
  }
}

@Component
struct SoundControl {
  @Link settings: NotificationSettings;

  build() {
    if (this.settings.enableSound) {
      Text('Sound On')
    } else {
      Text('Sound Off')
    }
  }
}

@Component
struct VibrationControl {
  @Link settings: NotificationSettings;

  build() {
    if (this.settings.enableSound) {
      Text('Vibration Enabled')
    } else {
      Text('Vibration Disabled')
    }
  }
}