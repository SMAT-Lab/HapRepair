@Entry
@Component
struct App {
  build() {
    ListItem()
      .onMouse(() => {
        const TAG = 'onMouse';
    })
  }
}