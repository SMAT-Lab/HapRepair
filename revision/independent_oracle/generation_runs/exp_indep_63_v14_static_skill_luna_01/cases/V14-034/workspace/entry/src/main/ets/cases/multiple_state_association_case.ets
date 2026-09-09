@Observed
class DashboardTheme {
  fontSize: number = 16;
  fontColor: string = '#222222';
}

@Component
struct DashboardTitle {
  @Link @Watch('onThemeChange') theme: DashboardTheme;
  @State localFontSize: number = 16;

  onThemeChange() {
    if (this.localFontSize !== this.theme.fontSize) {
      this.localFontSize = this.theme.fontSize;
    }
  }

  build() { Text('Title').fontSize(this.localFontSize) }
}

@Component
struct DashboardSubtitle {
  @Link @Watch('onThemeChange') theme: DashboardTheme;
  @State localFontColor: string = '#222222';

  onThemeChange() {
    if (this.localFontColor !== this.theme.fontColor) {
      this.localFontColor = this.theme.fontColor;
    }
  }

  build() { Text('Subtitle').fontColor(this.localFontColor) }
}

@Entry
@Component
struct DashboardPage {
  @State theme: DashboardTheme = new DashboardTheme();
  build() {
    Column() {
      DashboardTitle({ theme: this.theme })
      DashboardSubtitle({ theme: this.theme })
    }
  }
}
