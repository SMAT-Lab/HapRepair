@Observed
class ClassB {
  public name: string;
  constructor(name: string) {
    this.name = name;
  }
}

@Component
struct ItemComponent {
  @Prop itemData: ClassB;

  build() {
    Text(`Item name: ${this.itemData.name}`)
      .fontSize(18)
      .padding(10)
  }
}

@Entry
@Component
struct ListComponent {
  @State items: ClassB[] = [
    new ClassB('Item 1'),
    new ClassB('Item 2'),
    new ClassB('Item 3')
  ];

  build() {
    Column() {
      Text('Item List')
        .fontSize(24)
        .padding(10)

      ForEach(this.items, (item: ClassB) => {
        ItemComponent({ itemData: item })
      }, (item: ClassB) => item.name)
    }
    .padding(10)
  }
}
