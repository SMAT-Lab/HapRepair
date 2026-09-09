@Observed
class DashboardTheme {
  fontSize: number = 16;
  fontColor: string = '#222222';
}

@Component
struct DashboardTitle {
  @Link theme: DashboardTheme;
  build() { Text('Title').fontSize(this.theme.fontSize) }
}

@Component
struct DashboardSubtitle {
  @Link theme: DashboardTheme;
  build() { Text('Subtitle').fontColor(this.theme.fontColor) }
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
