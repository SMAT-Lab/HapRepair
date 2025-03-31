@Observed
class ThemeSettings {
  themeColor: string = 'light';
}

@Entry
@Component
struct DemoCFixed {
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
  @Link @Watch('onThemeChange') theme: ThemeSettings;
  @State localThemeColor: string = 'light';

  onThemeChange() {
    this.localThemeColor = this.theme.themeColor;
  }

  build() {
    Text('Header')
      .backgroundColor(this.localThemeColor === 'light' ? '#ffffff' : '#000000')
  }
}

@Component
struct Footer {
  @Link @Watch('onThemeChange') theme: ThemeSettings;
  @State localThemeColor: string = 'light';

  onThemeChange() {
    this.localThemeColor = this.theme.themeColor;
  }

  build() {
    Text('Footer')
      .backgroundColor(this.localThemeColor === 'light' ? '#ffffff' : '#000000')
  }
}