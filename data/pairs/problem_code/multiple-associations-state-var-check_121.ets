@Observed
class DisplaySettings {
  darkMode: boolean = false;
}

@Entry
@Component
struct DemoG {
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
  @Link display: DisplaySettings;

  build() {
    Text('Navbar')
      .backgroundColor(this.display.darkMode ? '#333333' : '#eeeeee')
  }
}

@Component
struct Sidebar {
  @Link display: DisplaySettings;

  build() {
    Text('Sidebar')
      .backgroundColor(this.display.darkMode ? '#333333' : '#eeeeee')
  }
}