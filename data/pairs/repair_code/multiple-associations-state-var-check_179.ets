@Observed
class NotificationSettings {
  enableSound: boolean = true;
}

@Entry
@Component
struct DemoEFixed {
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
  @Link @Watch('onSettingsChange') settings: NotificationSettings;
  @State localEnableSound: boolean = true;

  onSettingsChange() {
    this.localEnableSound = this.settings.enableSound;
  }

  build() {
    if (this.localEnableSound) {
      Text('Sound On')
    } else {
      Text('Sound Off')
    }
  }
}

@Component
struct VibrationControl {
  @Link @Watch('onSettingsChange') settings: NotificationSettings;
  @State localEnableSound: boolean = true;

  onSettingsChange() {
    this.localEnableSound = this.settings.enableSound;
  }

  build() {
    if (this.localEnableSound) {
      Text('Vibration Enabled')
    } else {
      Text('Vibration Disabled')
    }
  }
}