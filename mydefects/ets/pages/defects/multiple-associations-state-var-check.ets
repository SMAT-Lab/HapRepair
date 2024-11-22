@Observed
class SharedData {
  fontSize: number = 0;
  backgroundColor: string = '';
}

@Entry
@Component
struct MainComponent {
  @State sharedData: SharedData = new SharedData();
  private items: string[] = [];

  aboutToAppear(): void {
    for (let i = 0; i < 5; i++) {
      this.items.push(`Item ${i}`);
    }
  }

  build() {
    Column() {
      ForEach(this.items, (item: string, index: number) => {
        ListItemComponent({ item: item, index: index, sharedStyle: this.sharedData });
      }, (item: string) => item)

      Button('Change Font Size')
        .onClick(() => {
          this.sharedData.fontSize += 2;
        })
        .padding(10)

      Button('Change Background Color')
        .onClick(() => {
          this.sharedData.backgroundColor = this.sharedData.backgroundColor === 'red' ? 'blue' : 'red';
        })
        .padding(10)
    }
  }
}

@Component
struct ListItemComponent {
  @Prop item: string;
  @Prop index: number;
  @Link sharedStyle: SharedData;

  build() {
    Text(`${this.index}: ${this.item}`)
      .fontSize(this.sharedStyle.fontSize)
      .backgroundColor(this.sharedStyle.backgroundColor)
  }
}