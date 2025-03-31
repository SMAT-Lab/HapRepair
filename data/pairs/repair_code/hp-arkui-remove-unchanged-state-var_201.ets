class Position {
  x: number = 0;
  y: number = 0;
}

@Component
struct MyComponent {
  @State position: Position = new Position();
  // title 未发生变化，改为普通变量
  title: string = "Welcome";

  build() {
    Column() {
      Text(this.title)
        .fontSize(24)
      Button("Move")
        .onClick(() => {
          this.position.x += 10;
          this.position.y += 10;
        })
    }
    .position({
      x: this.position.x,
      y: this.position.y
    })
  }
}
