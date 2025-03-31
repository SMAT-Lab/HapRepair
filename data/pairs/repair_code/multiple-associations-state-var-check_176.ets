@Observed
class WindowStyle {
  isFullScreen: boolean = false;
}

@Entry
@Component
struct DemoBFixed {
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
  @Link @Watch('onStyleChange') style: WindowStyle;
  @State localFullScreen: boolean = false;

  onStyleChange() {
    this.localFullScreen = this.style.isFullScreen;
  }

  build() {
    if (this.localFullScreen) {
      Text('SideBar Hidden')
    } else {
      Text('SideBar Visible')
    }
  }
}

@Component
struct MainContent {
  @Link @Watch('onStyleChange') style: WindowStyle;
  @State localFullScreen: boolean = false;

  onStyleChange() {
    this.localFullScreen = this.style.isFullScreen;
  }

  build() {
    if (this.localFullScreen) {
      Text('FullScreen Mode')
    } else {
      Text('Normal Mode')
    }
  }
}