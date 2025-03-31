@Observed
class DisplaySettings {
  darkMode: boolean = false;
}

@Entry
@Component
struct DemoGFixed {
  @State display: DisplaySettings = new DisplaySettings();

  build() {
    Column() {
      Navbar({ display: this.display })
      Sidebar({ display: this.display })
      Text('Toggle Dark Mode')
        .onClick(() => {
          this.display.darkMode = !this.display.darkMode;
        })
    }
  }
}

@Component
struct Navbar {
  @Link @Watch('onModeChange') display: DisplaySettings;
  @State localDarkMode: boolean = false;

  onModeChange() {
    this.localDarkMode = this.display.darkMode;
  }

  build() {
    Text('Navbar')
      .backgroundColor(this.localDarkMode ? '#333333' : '#eeeeee')
  }
}

@Component
struct Sidebar {
  @Link @Watch('onModeChange') display: DisplaySettings;
  @State localDarkMode: boolean = false;

  onModeChange() {
    this.localDarkMode = this.display.darkMode;
  }

  build() {
    Text('Sidebar')
      .backgroundColor(this.localDarkMode ? '#333333' : '#eeeeee')
  }
}