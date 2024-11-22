@Entry
@Component
struct LargeListComponent {
  @State items: string[] = [
    'Apple', 'Banana', 'Orange', 'Grapes', 'Mango', 'Pineapple', 'Strawberry', 
    'Blueberry', 'Watermelon', 'Peach', 'Cherry', 'Papaya', 'Kiwi', 'Lychee', 
    'Pear', 'Plum', 'Apricot', 'Pomegranate', 'Dragonfruit', 'Cantaloupe'
  ];

  build() {
    List() {
      ForEach(this.items, (item: string) => {
        ListItem() {
          Text(item)
        }
      }, (item: string) => item)
    }
    .width('100%')
    .height('100%')
  }
}
