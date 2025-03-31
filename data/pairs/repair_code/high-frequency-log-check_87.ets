@Entry
@Component
struct UserInterface {
  build() {
    List()
      .onItemMove(() => {
        const EVENT = 'onItemMove';
      })
      .width("100%")
      .height("100%")
  }
}