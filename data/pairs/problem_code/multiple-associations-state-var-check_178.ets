@Observed
class ViewMode {
  displayMode: string = 'card';
}

@Entry
@Component
struct DemoD {
  @State view: ViewMode = new ViewMode();

  build() {
    Column() {
      ListView({ mode: this.view })
      GridView({ mode: this.view })
      Text('Switch Display Mode')
        .onClick(() => {
          this.view.displayMode = 'list';
        })
    }
  }
}

@Component
struct ListView {
  @Link mode: ViewMode;

  build() {
    Text('ListView')
      .visible(this.mode.displayMode === 'list')
  }
}

@Component
struct GridView {
  @Link mode: ViewMode;

  build() {
    Text('GridView')
      .visible(this.mode.displayMode === 'card')
  }
}