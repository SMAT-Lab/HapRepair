@Observed
class DashboardTheme {
  fontSize: number = 16;
  fontColor: string = '#222222';
}

@Component
struct DashboardTitle {
  @Link @Watch('syncTheme') theme: DashboardTheme;
  @State fontSize: number = 16;
  syncTheme(): void { this.fontSize = this.theme.fontSize; }
  build() { Text('Title').fontSize(this.fontSize) }
}

@Component
struct DashboardSubtitle {
  @Link @Watch('syncTheme') theme: DashboardTheme;
  @State fontColor: string = '#222222';
  syncTheme(): void { this.fontColor = this.theme.fontColor; }
  build() { Text('Subtitle').fontColor(this.fontColor) }
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
