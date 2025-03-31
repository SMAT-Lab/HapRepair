@Entry
@Component
struct App {
  build() {
    Toggle()
      .onActionUpdate(() => {
        const state = 'Toggle Action Update';
    })
  }
}