@Entry
@Component
struct CounterComponent {
  @State counter: number = 0;

  incrementCounter() {
    this.counter += 1;
    // 内部逻辑处理，不关联 UI
  }

  resetCounter() {
    this.counter = 0;
    // 内部逻辑处理，不关联 UI
  }

  build() {
    Column() {
      Button("开始")
        .onClick(() => this.doSomething())
      Button("停止")
        .onClick(() => this.doSomething())
      // `counter` 未在 UI 中使用，且未在 UI 调用的函数中使用
    }
  }

  doSomething() {
    // 一些无关的逻辑
  }
}
