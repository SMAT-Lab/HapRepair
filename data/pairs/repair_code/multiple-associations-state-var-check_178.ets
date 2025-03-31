@Observed
class ViewMode {
  displayMode: string = 'card';
}

@Entry
@Component
struct DemoDFixed {
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
  @Link @Watch('onModeChange') mode: ViewMode;
  @State localDisplayMode: string = 'card';

  onModeChange() {
    this.localDisplayMode = this.mode.displayMode;
  }

  build() {
    Text('ListView')
      .visible(this.localDisplayMode === 'list')
  }
}

@Component
struct GridView {
  @Link @Watch('onModeChange') mode: ViewMode;
  @State localDisplayMode: string = 'card';

  onModeChange() {
    this.localDisplayMode = this.mode.displayMode;
  }

  build() {
    Text('GridView')
      .visible(this.localDisplayMode === 'card')
  }
}