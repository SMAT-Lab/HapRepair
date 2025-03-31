@Component
struct GestureComponent {
  build() {
    Image('gesture.png')
      .gesture(TapGesture({ count: 2 })
        .onAction(() => {
          // 无业务逻辑
        })
      )
  }
}
