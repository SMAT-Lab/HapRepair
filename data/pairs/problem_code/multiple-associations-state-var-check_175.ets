@Observed
class UIConfig {
  layoutType: string = 'grid';
}

@Entry
@Component
struct DemoA {
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
  @Link layout: UIConfig;

  build() {
    Text('Header')
      .textAlign(this.layout.layoutType === 'grid' ? 'center' : 'left')
  }
}

@Component
struct FooterComp {
  @Link layout: UIConfig;

  build() {
    Text('Footer')
      .textAlign(this.layout.layoutType === 'grid' ? 'center' : 'left')
  }
}