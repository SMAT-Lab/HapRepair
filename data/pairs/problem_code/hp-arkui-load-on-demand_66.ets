@Entry
@Component
struct MyComponent {
  private items: number[] = Array.from({ length: 1000 }, (_, index) => index)

  build() {
    List() {
      ForEach(this.items, (item: number) => {
        ListItem() {
          Text(`Item: ${item}`)
        }
      })
    }
  }
}
