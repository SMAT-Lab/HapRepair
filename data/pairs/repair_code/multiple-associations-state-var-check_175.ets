@Observed
class UIConfig {
  layoutType: string = 'grid';
}

@Entry
@Component
struct DemoAFixed {
  @State config: UIConfig = new UIConfig();

  build() {
    Column() {
      HeaderComp({ layout: this.config })
      FooterComp({ layout: this.config })
      Text('Switch Layout')
        .onClick(() => {
          this.config.layoutType = 'list';
        })
    }
  }
}

@Component
struct HeaderComp {
  @Link @Watch('onLayoutChange') layout: UIConfig;
  @State localLayoutType: string = 'grid';

  onLayoutChange() {
    this.localLayoutType = this.layout.layoutType;
  }

  build() {
    Text('Header')
      .textAlign(this.localLayoutType === 'grid' ? 'center' : 'left')
  }
}

@Component
struct FooterComp {
  @Link @Watch('onLayoutChange') layout: UIConfig;
  @State localLayoutType: string = 'grid';

  onLayoutChange() {
    this.localLayoutType = this.layout.layoutType;
  }

  build() {
    Text('Footer')
      .textAlign(this.localLayoutType === 'grid' ? 'center' : 'left')
  }
}