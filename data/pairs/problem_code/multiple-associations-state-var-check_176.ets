@Observed
class WindowStyle {
  isFullScreen: boolean = false;
}

@Entry
@Component
struct DemoB {
  @State style: WindowStyle = new WindowStyle();

  build() {
    Row() {
      SideBar({ style: this.style })
      MainContent({ style: this.style })
      Text('Toggle Full Screen')
        .onClick(() => {
          this.style.isFullScreen = !this.style.isFullScreen;
        })
    }
  }
}

@Component
struct SideBar {
  @Link style: WindowStyle;

  build() {
    if (this.style.isFullScreen) {
      Text('SideBar Hidden')
    } else {
      Text('SideBar Visible')
    }
  }
}

@Component
struct MainContent {
  @Link style: WindowStyle;

  build() {
    if (this.style.isFullScreen) {
      Text('FullScreen Mode')
    } else {
      Text('Normal Mode')
    }
  }
}