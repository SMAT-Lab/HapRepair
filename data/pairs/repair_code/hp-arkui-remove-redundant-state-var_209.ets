@Entry
@Component
struct DataFetcher {
  data: any = null;

  fetchData() {
    // 从 API 获取数据并赋值给 `data`
    this.data = { /* 获取的数据 */ };
  }

  build() {
    Column() {
      Button("加载数据")
        .onClick(() => this.doSomething())
      // `data` 未在 UI 中使用，且未在 UI 调用的函数中使用
    }
  }

  doSomething() {
    // 一些无关的逻辑
  }
}
