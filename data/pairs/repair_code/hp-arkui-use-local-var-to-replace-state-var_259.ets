@Entry
@Component
struct CounterComponent {
  @State count: number = 0;

  increment() {
    let newCount = this.count;
    newCount += 1;
    newCount += 1;
    newCount += 1;
    this.count = newCount;
  }

  build() {
    Column() {
      Text('计数值：' + this.count)
        .fontSize(20)
      Button('增加计数')
        .onClick(() => this.increment())
    }
  }
}
