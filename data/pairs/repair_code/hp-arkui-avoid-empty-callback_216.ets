@Component
struct TextInputComponent {
  handleInputChange(value: string) {
    // 处理输入变化的逻辑
  }

  build() {
    TextInput()
      .onChange((value) => {
        this.handleInputChange(value)
      })
  }
}
