@Entry
@Component
struct Home {
  build() {
    TextInput()
      .onTextChange(() => {
        const TAG = 'onTextChange';
      })
  }
}