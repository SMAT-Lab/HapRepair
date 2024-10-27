@Component
struct TestComponent {
  build() {
    Slider({ min: 0, max: 100 })
      .onChange(() => {
        // 无业务逻辑
      })
  }
}