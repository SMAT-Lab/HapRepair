@Entry
@Component
struct CounterComponent {
  @State counter: number = 0;

  // 每次递增时直接更新状态变量，而不是使用临时变量
  incrementMultipleTimes() {
    for (let i = 0; i < 5; i++) {
      this.counter += 1;
      this.counter += 2;
      this.counter += 3;
    }
  }

  build() {
    Column() {
      Text('Counter: ' + this.counter)  // 显示计数器值
        .fontSize(20)
        .padding(10)

      Button('Increase Counter')
        .onClick(() => {
          // 点击时递增计数器值，直接更新状态变量
          this.incrementMultipleTimes();
        })
        .fontSize(16)
        .backgroundColor(Color.Gray)
        .padding(10)
    }
    .padding(20)
  }
}