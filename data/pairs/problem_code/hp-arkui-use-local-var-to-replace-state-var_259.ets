@Entry
@Component
struct CounterComponent {
  @State count: number = 0;

  increment() {
    this.count += 1;
    this.count += 1;
    this.count += 1;
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
