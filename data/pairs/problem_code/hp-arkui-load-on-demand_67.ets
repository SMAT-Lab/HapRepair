@Entry
@Component
struct MyComponent {
  private items: string[] = ["Apple", "Banana", "Cherry", "Date"]

  build() {
    List() {
      ForEach(this.items, (item: string) => {
        ListItem() {
          Text(item)
        }
      })
    }
  }
}
