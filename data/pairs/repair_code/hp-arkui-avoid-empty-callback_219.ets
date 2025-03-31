@Component
struct GestureComponent {
  handleDoubleTap() {
    // 处理双击手势的逻辑
  }

  build() {
    Image('gesture.png')
      .gesture(TapGesture({ count: 2 })
        .onAction(() => {
          this.handleDoubleTap()
        })
      )
  }
}
