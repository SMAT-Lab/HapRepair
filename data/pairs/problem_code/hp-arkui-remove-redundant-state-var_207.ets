@Entry
@Component
struct LogComponent {
  @State logMessages: string[] = [];

  addLog(message: string) {
    this.logMessages.push(message);
    // 执行日志记录操作
  }

  doSomething() {
    // 一些逻辑处理
  }

  build() {
    Column() {
      Button("执行操作")
        .onClick(() => this.doSomething())
      // `logMessages` 未在 UI 中使用，且未在 UI 调用的函数中使用
    }
  }
}
