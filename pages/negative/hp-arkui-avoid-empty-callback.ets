// 避免设置空的系统回调监听。
//
// 根据ArkUI编程规范，建议修改。

@Component
struct MyComponent {
  build() {
    Button('Click', { type: ButtonType.Normal, stateEffect: true })
      .onClick(() => {
        // 无业务逻辑
      })
  }
}