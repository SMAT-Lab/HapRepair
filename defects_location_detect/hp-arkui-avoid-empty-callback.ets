@Component
struct TestComponent {
  build() {
    Slider({ min: 0, max: 100 })
      .onChange(() => {

      })
  }
}
