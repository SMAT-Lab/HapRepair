@Observed
class ThemeSettings {
  themeColor: string = 'light';
}

@Entry
@Component
struct DemoC {
  @State theme: ThemeSettings = new ThemeSettings();

  build() {
    Column() {
      Header({ theme: this.theme })
      Footer({ theme: this.theme })
      Text('Switch Theme')
        .onClick(() => {
          this.theme.themeColor = 'dark';
        })
    }
  }
}

@Component
struct Header {
  @Link theme: ThemeSettings;

  build() {
    Text('Header')
      .backgroundColor(this.theme.themeColor === 'light' ? '#ffffff' : '#000000')
  }
}

@Component
struct Footer {
  @Link theme: ThemeSettings;

  build() {
    Text('Footer')
      .backgroundColor(this.theme.themeColor === 'light' ? '#ffffff' : '#000000')
  }
}